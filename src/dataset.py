import fcntl
import shutil
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from pytorchvideo.data.encoded_video import EncodedVideo

from torchvision.transforms import v2


# Truncated in the AUTSL release
CORRUPT_CLIPS = {
    "signer15_sample276_color.mp4",
    "signer20_sample424_color.mp4",
    "signer22_sample612_color.mp4",
    "signer2_sample1161_color.mp4",
    "signer36_sample356_color.mp4",
    "signer3_sample152_color.mp4",
    "signer41_sample25_color.mp4",
    "signer42_sample647_color.mp4",
    "signer5_sample618_color.mp4",
    "signer8_sample1406_color.mp4",
    "signer6_sample185_color.mp4",
}


class UniformTemporalSubsample:
    """
        Pick num_samples frames evenly spaced along the temporal axis
    """

    def __init__(self, num_samples):
        self.num_samples = num_samples

    def __call__(self, video):
        # Clips arrive as [C, T, H, W], so the temporal axis is -3
        frames = video.shape[-3]

        # Frames are repeated by nearest neighbour if the clip is too short
        indices = torch.linspace(0, frames - 1, self.num_samples)

        return video.index_select(-3, indices.clamp(0, frames - 1).long())


# Module level so DataLoader workers can pickle them (local lambdas cannot)
def _to_unit_float(video):
    # uint8 [T, C, H, W] -> float [T, C, H, W] in [0, 1]
    return video.float() / 255.0


def _to_x3d_layout(video):
    # [T, C, H, W] -> [C, T, H, W]
    return video.permute(1, 0, 2, 3)


class _DecodeView(Dataset):
    """
        The decode step alone, so DataLoader workers can fill the cache in parallel
    """

    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset.samples)

    def __getitem__(self, index):
        return self.dataset._decode(index)


# AUTSL sign language clips: one .mp4 per sample, 226 word classes
class AUTSLDataset(Dataset):
    def __init__(
        self,
        annotation_file,
        video_root,
        num_frames = 4,
        crop_size = 160,
        train = False,
        subset = None,
        cache_dir = None,
        num_workers = 0,
    ):
        # num_frames and crop_size should match the X3D variant being trained
        self.video_root = Path(video_root)
        self.num_frames = num_frames
        self.crop_size = crop_size
        self.train = train

        # Read annotations
        self.samples = []

        with open(annotation_file, "r") as f:
            for line in f:
                # Each row is "<video_id>,<label>", e.g. "signer0_sample1,41"
                video_id, label = line.strip().split(",")

                # On disk the clip is named "<video_id>_color.mp4"
                clip = f"{video_id}_color.mp4"

                if clip in CORRUPT_CLIPS:
                    continue

                self.samples.append((clip, int(label)))

        # Drawn before the cache is built, so only the kept clips are decoded
        if subset:
            keep = torch.randperm(len(self.samples))[:subset].tolist()
            self.samples = [self.samples[i] for i in keep]

        # Deterministic half of the preprocessing, applied once per clip
        self.subsample = UniformTemporalSubsample(num_frames)
        self.resize = v2.Resize((crop_size, crop_size))

        # Stochastic and cheap half, applied every epoch
        self.transform = self._build_transform()

        # Mapped lazily by __getitem__, so each worker opens it for itself
        self.cache_shape = (len(self.samples), num_frames, 3, crop_size, crop_size)
        self.cache_path = None
        self.cache = None

        if cache_dir is not None:
            name = f"{self.video_root.name}_{num_frames}x{crop_size}.u8"
            self.cache_path = Path(cache_dir) / name

            self._build_cache(num_workers)


    def _build_transform(self):
        # Runs on uint8 [T, C, H, W] frames, whether cached or freshly decoded

        transforms = [
            # v2 reads the last three dims as [C, H, W] and expects [0, 1]
            v2.Lambda(_to_unit_float),
        ]

        if self.train:
            # Augmentation only on training, so it stays out of the cache
            transforms.append(
                v2.RandomHorizontalFlip(p = 0.5)
            )

        transforms.extend([
            # Kinetics statistics, matching what the pretrained weights expect
            v2.Normalize(
                mean = (0.45, 0.45, 0.45),
                std = (0.225, 0.225, 0.225)
            ),

            # Back to [C, T, H, W] for X3D
            v2.Lambda(_to_x3d_layout),
        ])

        return v2.Compose(transforms)


    def _decode(self, index):
        """
            Decode one clip to uint8 [T, C, H, W] at the variant's geometry

            Nothing here depends on the epoch, which is what makes it cacheable.
        """
        clip, _ = self.samples[index]

        # Decode lazily, one clip per call
        video = EncodedVideo.from_path(
            str(self.video_root / clip),
            decode_audio = False
        )

        # AUTSL clips are ~2s, so take the whole video and subsample from it
        video_data = video.get_clip(
            start_sec = 0,
            end_sec = video.duration
        )

        # Decoder gives [C, T, H, W] with integral values in [0, 255]
        frames = self.subsample(video_data["video"])

        return self.resize(frames.permute(1, 0, 2, 3).to(torch.uint8))


    def _cached(self, marker):
        # The shape is written only once the array has been filled completely
        return marker.is_file() and marker.read_text() == str(self.cache_shape)


    def _build_cache(self, num_workers):
        """
            Decode every clip once into a memory-mapped uint8 array

            The training loop reads the same clips on every epoch, so over a long
            run the decode dominates. Paying it once here turns each epoch into a
            sequence of reads. One array task builds while the others wait on the
            lock, then find the marker and read what it wrote.
        """
        marker = self.cache_path.with_suffix(".done")

        self.cache_path.parent.mkdir(parents = True, exist_ok = True)

        if self._cached(marker):
            print(f"cache hit: {self.cache_path}", flush = True)
            return

        with open(self.cache_path.with_suffix(".lock"), "w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)

            # Whoever held the lock may have just built it
            if self._cached(marker):
                print(f"cache hit: {self.cache_path}", flush = True)
                return

            self._fill_cache(marker, num_workers)


    def _fill_cache(self, marker, num_workers):
        """
            Decode into the memory map, called with the build lock held
        """
        needed = int(np.prod(self.cache_shape))
        free = shutil.disk_usage(self.cache_path.parent).free

        print(
            f"building cache: {self.cache_path} "
            f"({len(self.samples)} clips, {needed / 1e9:.1f} GB needed, "
            f"{free / 1e9:.1f} GB free)",
            flush = True,
        )

        # Running a memory map past the end of its filesystem raises SIGBUS half
        # way through the build, so refuse before decoding anything
        if free < needed:
            raise RuntimeError(
                f"{self.cache_path.parent} has {free / 1e9:.1f} GB free, but the "
                f"cache needs {needed / 1e9:.1f} GB. Set CACHE_DIR to a "
                f"filesystem with room, and note that a tmpfs such as /tmp is "
                f"charged against the job's --mem."
            )

        # A stale marker must not survive a failed rebuild
        marker.unlink(missing_ok = True)

        # Batched only to amortise the worker handover; order is preserved
        loader = DataLoader(
            _DecodeView(self),
            batch_size = 16,
            shuffle = False,
            num_workers = num_workers,
        )

        filled = 0

        # Written through the file rather than the memory map it is read back
        # through: a map that meets a full disk or an exhausted quota raises
        # SIGBUS and dumps core, whereas a write reports the error and unwinds
        try:
            with open(self.cache_path, "wb") as cache:
                for batch in loader:
                    cache.write(batch.numpy().tobytes())
                    filled += len(batch)
        except OSError:
            # Half a cache is worth nothing, and the room may be wanted back
            self.cache_path.unlink(missing_ok = True)
            raise

        marker.write_text(str(self.cache_shape))

        print(f"cached {filled} clips", flush = True)


    def __len__(self):
        return len(self.samples)


    def __getitem__(self, index):

        _, label = self.samples[index]

        if self.cache_path is None:
            frames = self._decode(index)
        else:
            if self.cache is None:
                self.cache = np.memmap(
                    self.cache_path,
                    dtype = np.uint8,
                    mode = "r",
                    shape = self.cache_shape,
                )

            # Copied out, since a read-only map cannot back a torch tensor
            frames = torch.from_numpy(np.array(self.cache[index]))

        # [C, T, H, W]
        video = self.transform(frames)

        return video, torch.tensor(label, dtype=torch.long)

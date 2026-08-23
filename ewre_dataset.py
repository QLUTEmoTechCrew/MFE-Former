import csv
import re
from dataclasses import dataclass
from pathlib import Path

import torch
import torchaudio
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Dataset
from torch.utils.data._utils.collate import default_collate


EWRE_FILENAME = re.compile(
    r"^(?P<score>\d+)_(?P<label>[01])_(?P<subject>\d{3})_(?P<word>\d+)\.wav$",
    re.IGNORECASE,
)
SEX_MAP = {"男": "male", "女": "female"}
EDUCATION_MAP = {
    "小学": "primary school",
    "初中": "middle school",
    "高中": "high school",
    "大学": "college",
    "研究生": "graduate",
    "学生": "student",
}
EDUCATION_LEVELS = tuple(EDUCATION_MAP)


@dataclass(frozen=True)
class ParsedEWREName:
    score: int
    label: int
    subject_id: str
    subject_number: str
    word_index: int


@dataclass(frozen=True)
class SubjectRecord:
    subject_id: str
    subject_number: str
    score: int
    label: int
    word_paths: tuple
    age: float = 0.0
    sex: str = "unknown"
    education: str = "unknown"

    @property
    def identity_text(self):
        return (
            f"a spectrogram of a {self.age:g} years old {self.sex} "
            f"who has a {self.education} education"
        )

    @property
    def numeric_identity(self):
        sex_vector = [float(self.sex == "male"), float(self.sex == "female")]
        education_vector = [
            float(self.education == EDUCATION_MAP[level]) for level in EDUCATION_LEVELS
        ]
        return [self.age / 100.0, *sex_vector, *education_vector]


@dataclass(frozen=True)
class FoldSplit:
    train_ids: tuple
    test_ids: tuple


def parse_ewre_filename(filename):
    match = EWRE_FILENAME.fullmatch(Path(filename).name)
    if match is None:
        raise ValueError(
            f"Invalid EWRE filename {filename!r}; expected <score>_<label>_<subject>_<word>.wav."
        )
    score = int(match.group("score"))
    label = int(match.group("label"))
    subject_number = match.group("subject")
    word_index = int(match.group("word"))
    return ParsedEWREName(
        score=score,
        label=label,
        subject_id=f"{match.group('score')}_{label}_{subject_number}",
        subject_number=subject_number,
        word_index=word_index,
    )


def _read_metadata_file(path, label):
    records = {}
    with Path(path).open("r", encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            participant_id = row["编号"].strip()
            subject_number = participant_id[1:].zfill(3)
            records[(label, subject_number)] = {
                "age": float(row["年龄"]),
                "sex": SEX_MAP.get(row["性别"].strip(), row["性别"].strip()),
                "education": EDUCATION_MAP.get(
                    row["教育程度"].strip(),
                    row["教育程度"].strip(),
                ),
                "score": int(float(row["总分"])),
            }
    return records


def _read_combined_metadata(path):
    records = {}
    with Path(path).open("r", encoding="gb18030", newline="") as stream:
        for row in csv.DictReader(stream):
            participant_id = row["id"].strip()
            label = int(row["label"])
            subject_number = participant_id[1:].zfill(3)
            records[(label, subject_number)] = {
                "age": float(row["age"]),
                "sex": SEX_MAP.get(row["sex"].strip(), row["sex"].strip()),
                "education": EDUCATION_MAP.get(row["edu"].strip(), row["edu"].strip()),
                "score": int(float(row["score"])),
            }
    return records


def load_ewre_metadata(data_root):
    data_root = Path(data_root)
    combined_path = data_root / "EWRE.csv"
    if combined_path.is_file():
        return _read_combined_metadata(combined_path)
    metadata = {}
    metadata.update(_read_metadata_file(data_root / "normal.csv", label=0))
    metadata.update(_read_metadata_file(data_root / "depression.csv", label=1))
    return metadata


def index_ewre_subjects(data_root, expected_words=72):
    data_root = Path(data_root)
    wav_dir = data_root / "audio" / "wav"
    if not wav_dir.is_dir():
        raise FileNotFoundError(f"EWRE WAV directory not found: {wav_dir}")
    metadata = load_ewre_metadata(data_root)
    grouped = {}
    descriptors = {}
    for path in sorted(wav_dir.glob("*.wav")):
        parsed = parse_ewre_filename(path.name)
        grouped.setdefault(parsed.subject_id, {})
        if parsed.word_index in grouped[parsed.subject_id]:
            raise ValueError(f"Duplicate word {parsed.word_index} for {parsed.subject_id}.")
        grouped[parsed.subject_id][parsed.word_index] = path
        descriptors[parsed.subject_id] = parsed

    records = []
    expected_indices = set(range(1, expected_words + 1))
    for subject_id in sorted(grouped):
        parsed = descriptors[subject_id]
        observed_indices = set(grouped[subject_id])
        if observed_indices != expected_indices:
            missing = sorted(expected_indices - observed_indices)
            extra = sorted(observed_indices - expected_indices)
            raise ValueError(f"{subject_id} has invalid word indices; missing={missing}, extra={extra}.")
        metadata_key = (parsed.label, parsed.subject_number)
        if metadata_key not in metadata:
            raise ValueError(f"Missing demographics for {subject_id}.")
        participant = metadata[metadata_key]
        if participant["score"] != parsed.score:
            raise ValueError(
                f"HAMD score mismatch for {subject_id}: filename={parsed.score}, metadata={participant['score']}."
            )
        records.append(
            SubjectRecord(
                subject_id=subject_id,
                subject_number=parsed.subject_number,
                score=parsed.score,
                label=parsed.label,
                word_paths=tuple(grouped[subject_id][index] for index in range(1, expected_words + 1)),
                age=participant["age"],
                sex=participant["sex"],
                education=participant["education"],
            )
        )
    if not records:
        raise ValueError(f"No EWRE WAV files found under {wav_dir}.")
    return records


def build_stratified_folds(subjects, n_splits=5, seed=42):
    subject_ids = [record.subject_id for record in subjects]
    labels = [record.label for record in subjects]
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = []
    for train_index, test_index in splitter.split(subject_ids, labels):
        folds.append(
            FoldSplit(
                train_ids=tuple(subject_ids[index] for index in train_index),
                test_ids=tuple(subject_ids[index] for index in test_index),
            )
        )
    return folds


class EWREDataset(Dataset):
    def __init__(
        self,
        records,
        sample_rate=48000,
        n_mels=80,
        window_ms=25.0,
        hop_ms=10.0,
        max_frames=832,
        normalize=False,
        clip_features=None,
        cache_audio=True,
    ):
        self.records = list(records)
        self.sample_rate = sample_rate
        self.max_frames = max_frames
        self.normalize = normalize
        self.clip_features = clip_features
        self.cache_audio = cache_audio
        self._audio_cache = {}
        win_length = round(sample_rate * window_ms / 1000.0)
        hop_length = round(sample_rate * hop_ms / 1000.0)
        n_fft = 1 << (win_length - 1).bit_length()
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            n_mels=n_mels,
            power=2.0,
        )
        self.db_transform = torchaudio.transforms.AmplitudeToDB(stype="power", top_db=80.0)

    def __len__(self):
        return len(self.records)

    def _load_word(self, path):
        waveform, source_rate = torchaudio.load(path)
        waveform = waveform.mean(dim=0, keepdim=True)
        if source_rate != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, source_rate, self.sample_rate)
        mel = self.db_transform(self.mel_transform(waveform)).squeeze(0).transpose(0, 1)
        mel = mel[: self.max_frames]
        if self.normalize:
            mel = (mel - mel.mean()) / mel.std().clamp_min(1e-6)
        valid_frames = mel.size(0)
        padded = mel.new_zeros(self.max_frames, mel.size(1))
        padded[:valid_frames] = mel
        mask = torch.zeros(self.max_frames, dtype=torch.bool)
        mask[:valid_frames] = True
        return padded, mask

    def __getitem__(self, index):
        record = self.records[index]
        if index in self._audio_cache:
            cached_speech, frame_mask = self._audio_cache[index]
            speech = cached_speech.float()
        else:
            words = [self._load_word(path) for path in record.word_paths]
            speech = torch.stack([item[0] for item in words])
            frame_mask = torch.stack([item[1] for item in words])
            if self.cache_audio:
                self._audio_cache[index] = (speech.half(), frame_mask)
        item = {
            "speech": speech,
            "frame_mask": frame_mask,
            "label": torch.tensor(record.label, dtype=torch.long),
            "identity": torch.tensor(record.numeric_identity, dtype=torch.float32),
            "identity_text": record.identity_text,
            "subject_id": record.subject_id,
        }
        if self.clip_features is not None:
            item["clip_text_features"] = self.clip_features[record.subject_id]
        return item


def collate_ewre_batch(batch):
    """Collate fixed-size cached samples, then remove batch-wide zero padding."""
    collated = default_collate(batch)
    max_valid_frames = int(collated["frame_mask"].sum(dim=-1).max().item())
    collated["speech"] = collated["speech"][:, :, :max_valid_frames]
    collated["frame_mask"] = collated["frame_mask"][:, :, :max_valid_frames]
    return collated

from ewre_dataset import SubjectRecord, build_stratified_folds, parse_ewre_filename


def test_parse_ewre_filename():
    item = parse_ewre_filename("36_1_001_72.wav")

    assert item.score == 36
    assert item.label == 1
    assert item.subject_id == "36_1_001"
    assert item.subject_number == "001"
    assert item.word_index == 72


def test_subject_folds_have_no_overlap():
    subjects = [
        SubjectRecord(
            subject_id=f"{index:02d}_{label}_{index:03d}",
            subject_number=f"{index:03d}",
            score=index,
            label=label,
            word_paths=(),
        )
        for label in (0, 1)
        for index in range(10)
    ]

    folds = build_stratified_folds(subjects, n_splits=5, seed=42)

    assert len(folds) == 5
    for fold in folds:
        assert set(fold.train_ids).isdisjoint(fold.test_ids)
        assert len(fold.test_ids) == 4

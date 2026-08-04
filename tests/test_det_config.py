"""검출 config 생성 불변 규칙 — 경로 규약과 Colab 세션 대비 설정."""

import yaml

from src.configs.build_det_config import build, deep_merge, det_overrides

KW = dict(root="/content/label_ocr", weights="/w/best_accuracy",
          save_dir="/drive/exp/det", epochs=30, batch_size=4)


def test_deep_merge_keeps_untouched_keys():
    base = {"Global": {"a": 1, "b": 2}, "Other": {"x": 1}}
    got = deep_merge(base, {"Global": {"b": 9}})
    assert got == {"Global": {"a": 1, "b": 9}, "Other": {"x": 1}}


def test_deep_merge_does_not_mutate_base():
    base = {"Global": {"a": 1}}
    deep_merge(base, {"Global": {"a": 2}})
    assert base == {"Global": {"a": 1}}


def test_data_dir_points_at_image_set():
    """라벨의 이미지 경로가 디렉터리 접두사 없는 파일명이라 data_dir 가 image_set 이어야 한다."""
    o = det_overrides(**KW)
    assert o["Train"]["dataset"]["data_dir"].endswith("/image_set")
    assert o["Eval"]["dataset"]["data_dir"] == o["Train"]["dataset"]["data_dir"]


def test_train_and_eval_use_different_splits():
    """같은 split 을 학습·평가에 쓰면 성능이 과대평가된다."""
    o = det_overrides(**KW)
    assert o["Train"]["dataset"]["label_file_list"] != o["Eval"]["dataset"]["label_file_list"]
    assert o["Eval"]["dataset"]["label_file_list"][0].endswith("det_val.txt")


def test_test_split_is_never_referenced():
    """test 는 최종 평가 전용 — 학습 config 에 등장하면 안 된다."""
    assert "det_test.txt" not in str(det_overrides(**KW))


def test_checkpoints_are_saved_often_for_colab():
    """세션이 끊겨도 이어서 학습하려면 자주 저장하고 resume 가능해야 한다."""
    o = det_overrides(**KW)["Global"]
    assert o["save_epoch_step"] <= 5
    assert "checkpoints" in o


def test_resume_checkpoint_is_propagated():
    assert det_overrides(**{**KW, "checkpoints": "/drive/exp/det/latest"})["Global"]["checkpoints"] \
        == "/drive/exp/det/latest"


def test_amp_is_on_by_default():
    """Colab VRAM 여유 확보 (지시서 6-C)."""
    assert det_overrides(**KW)["Global"]["use_amp"] is True


def test_build_merges_into_a_real_template(tmp_path):
    template = tmp_path / "t.yml"
    template.write_text(yaml.safe_dump({
        "Global": {"use_gpu": False, "epoch_num": 500, "debug": False},
        "Architecture": {"Backbone": {"name": "PPHGNet_small"}},
        "PostProcess": {"name": "DBPostProcess", "thresh": 0.3, "box_thresh": 0.6},
        "Train": {"dataset": {"name": "SimpleDataSet", "data_dir": "./x",
                              "transforms": [{"DecodeImage": {}}]},
                  "loader": {"batch_size_per_card": 8}},
        "Eval": {"dataset": {"name": "SimpleDataSet", "data_dir": "./x"},
                 "loader": {"batch_size_per_card": 1}},
    }), encoding="utf-8")
    out = tmp_path / "out.yml"
    cfg = build(str(template), str(out), **KW)

    assert cfg["Global"]["use_gpu"] is True and cfg["Global"]["epoch_num"] == 30
    assert cfg["Architecture"]["Backbone"]["name"] == "PPHGNet_small"   # 템플릿 보존
    assert cfg["Train"]["dataset"]["transforms"] == [{"DecodeImage": {}}]
    assert cfg["Train"]["loader"]["batch_size_per_card"] == 4
    assert yaml.safe_load(out.read_text(encoding="utf-8")) == cfg


def test_postprocess_defaults_are_untouched(tmp_path):
    """M1 스윕에서 기본값을 넘는 조합이 없었으므로 건드리지 않는다."""
    assert "PostProcess" not in det_overrides(**KW)

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


def test_amp_is_off_by_default():
    """AMP 는 DBLoss 의 Dice 항을 조용히 망가뜨린다 — 켜면 학습이 통째로 무의미해진다.

    fp16 에서 `union = sum(pred * mask)` 가 164만 픽셀 합이라 상한 65504 를 넘어 inf 가
    되고, intersection 은 텍스트 픽셀(약 2%)만 더해 정상이므로 Dice 가 정확히 1.0 에
    고정된다(2026-08-04 A100 실측: loss_shrink_maps 가 300스텝 내내 5.000000).
    순전파 출력은 정상으로 보이므로 이 회귀는 손실값으로만 잡힌다.
    VRAM 절감보다 학습이 실제로 되는 것이 우선이다.
    """
    assert det_overrides(**KW)["Global"]["use_amp"] is False


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


def test_learning_rate_is_lowered_for_finetuning():
    """템플릿 기본 1e-3 은 10만 장급 코퍼스용이라 1,872장 fine-tuning 에는 과하다.

    실측(2026-08-04 A100): 1e-3 으로 6 epoch 학습하면 val hmean 이
    0.6370 -> 0.6024 로 **내려간다**. 사전학습 특징이 파괴되는 것이다.
    가르칠 것은 글자 검출 자체가 아니라 박스 규약 하나뿐이다.
    """
    assert det_overrides(**KW)["Optimizer"]["lr"]["learning_rate"] == 1e-4
    assert det_overrides(**{**KW, "lr": 5e-5})["Optimizer"]["lr"]["learning_rate"] == 5e-5

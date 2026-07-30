"""박스 폴리곤 → 인식용 크롭.

Phase 2 게이트 실측 결과 (곡면 99박스, PP-OCRv5_server_rec 로 CAR 비교):

    방식                              전체 CAR   빈 출력
    poly (PaddleX segment unrolling)   0.4387    27/99
    poly + 점순서 정규화               0.4409    22/99
    quad (minAreaRect)                 0.6172     0/99

지시서가 "권장"한 segment unrolling(`CropByPolys(det_box_type="poly")`)은
5점 폴리곤에서는 minAreaRect 와 결과가 동일하지만, **6점 이상에서 기하가 무너진다**
(종횡비가 3.4 → 0.97 처럼 뒤집히고, 인식기가 빈 문자열을 뱉는다). 점 순서를
시계방향·좌상단 시작으로 정규화해도 해소되지 않았다. 따라서 지시서의 **대안 경로인
최소외접사각형 폴백을 채택**하고, 곡률 보정은 인식기의 rectification 에 맡긴다
(Phase 0 에서 `rec_svtrnet.yml` 에 STN_ON 이 내장됨을 확인 — docs/framework_decision.md §6).

4점 박스는 두 방식이 동등했으므로(CAR 0.7922 vs 0.7902, 표본 300) 지시서대로
정확한 4점 원근변환(`get_rotate_crop_image`)을 쓴다.
"""

import numpy as np
from paddlex.inference.pipelines.components.common.crop_image_regions import CropByPolys

_op = CropByPolys(det_box_type="quad")


def crop_box(img: np.ndarray, points) -> np.ndarray:
    """박스 하나를 크롭한다. 4점은 원근변환, 5점 이상은 최소외접사각형."""
    pts = np.array(points, dtype=np.float32)
    if len(pts) == 4:
        return _op.get_rotate_crop_image(img, pts.copy())
    return _op.get_minarea_rect_crop(img, pts.copy())

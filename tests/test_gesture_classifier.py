from __future__ import annotations

import unittest
from dataclasses import dataclass

from app import GestureSmoother, HandGestureDetector


@dataclass
class Point:
    x: float
    y: float


def base_closed_hand() -> list[Point]:
    points = [Point(0.5, 0.8) for _ in range(21)]
    points[0] = Point(0.50, 0.80)
    points[1] = Point(0.47, 0.70)
    points[2] = Point(0.46, 0.60)
    points[3] = Point(0.44, 0.62)
    points[4] = Point(0.43, 0.66)
    points[5] = Point(0.48, 0.56)
    points[6] = Point(0.48, 0.58)
    points[7] = Point(0.46, 0.62)
    points[8] = Point(0.44, 0.65)
    points[9] = Point(0.50, 0.54)

    for mcp, pip, dip, tip, x in [
        (9, 10, 11, 12, 0.50),
        (13, 14, 15, 16, 0.53),
        (17, 18, 19, 20, 0.56),
    ]:
        points[mcp] = Point(x, 0.56)
        points[pip] = Point(x, 0.60)
        points[dip] = Point(x, 0.66)
        points[tip] = Point(x, 0.72)

    return points


def raise_finger(points: list[Point], pip_id: int, dip_id: int, tip_id: int) -> None:
    points[pip_id].y = 0.50
    points[dip_id].y = 0.42
    points[tip_id].y = 0.34


class GestureClassifierTest(unittest.TestCase):
    def classify(self, points: list[Point]) -> str:
        return HandGestureDetector._classify_gesture(points, "Right")

    def test_fist_is_zero_not_bite_or_thumb(self) -> None:
        self.assertEqual(self.classify(base_closed_hand()), "zero")

    def test_spread_thumb_plus_two_fingers_counts_as_three(self) -> None:
        points = base_closed_hand()
        raise_finger(points, 6, 7, 8)
        raise_finger(points, 10, 11, 12)
        points[2] = Point(0.44, 0.62)
        points[3] = Point(0.32, 0.64)
        points[4] = Point(0.22, 0.66)

        self.assertEqual(self.classify(points), "three")

    def test_bite_requires_pinch_outside_closed_fist(self) -> None:
        points = base_closed_hand()
        points[4] = Point(0.46, 0.42)
        points[8] = Point(0.48, 0.40)

        self.assertEqual(self.classify(points), "bite")

    def test_four_and_five_split_on_thumb_spread(self) -> None:
        four = base_closed_hand()
        for pip_id, dip_id, tip_id in [(6, 7, 8), (10, 11, 12), (14, 15, 16), (18, 19, 20)]:
            raise_finger(four, pip_id, dip_id, tip_id)

        five = [Point(point.x, point.y) for point in four]
        five[2] = Point(0.44, 0.62)
        five[3] = Point(0.32, 0.64)
        five[4] = Point(0.22, 0.66)

        self.assertEqual(self.classify(four), "four")
        self.assertEqual(self.classify(five), "five")


class GestureSmootherTest(unittest.TestCase):
    def test_switches_only_after_confirmed_frames(self) -> None:
        smoother = GestureSmoother(confirm_frames=3)

        self.assertEqual(smoother.update("two"), "no_hand")
        self.assertEqual(smoother.update("two"), "no_hand")
        self.assertEqual(smoother.update("three"), "no_hand")
        self.assertEqual(smoother.update("three"), "no_hand")
        self.assertEqual(smoother.update("three"), "three")


if __name__ == "__main__":
    unittest.main()

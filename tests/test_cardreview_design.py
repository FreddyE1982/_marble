"""Unit tests for the CardReview design helpers."""

import unittest

from cardreview.design import (
    CardDesign,
    CardDimensions,
    CardFont,
    CardPalette,
    ColorStyle,
    ImageElement,
    TextElement,
)


class TestCardDesignToolkit(unittest.TestCase):
    """Validate the card design features exposed to the CardReview app."""

    def setUp(self) -> None:
        self.dimensions = CardDimensions(width_mm=88.0, height_mm=63.0)
        self.palette = CardPalette(
            background=ColorStyle(red=240, green=242, blue=245),
            accent=ColorStyle(red=30, green=102, blue=220),
            text=ColorStyle(red=20, green=23, blue=31),
        )
        self.design = CardDesign(dimensions=self.dimensions, palette=self.palette)
        print("Test setup palette:", self.palette)

    def test_text_alignment_controls(self) -> None:
        font = CardFont(family="Inter", size_pt=16.0, weight="regular")
        prompt = TextElement(
            x_mm=10.0,
            y_mm=20.0,
            width_mm=60.0,
            height_mm=18.0,
            text="Prompt",
            font=font,
        )
        prompt.change_alignment("center")
        self.design.front.elements.append(prompt)
        self.design.front.align_center()
        print("Front element alignment:", prompt.alignment, prompt.x_mm)
        self.assertEqual(prompt.alignment, "center")
        self.assertAlmostEqual(prompt.x_mm, 0.0)

    def test_vertical_distribution_orders_elements(self) -> None:
        font = CardFont(family="Inter", size_pt=14.0, weight="regular")
        first = TextElement(
            x_mm=12.0,
            y_mm=30.0,
            width_mm=50.0,
            height_mm=10.0,
            text="First",
            font=font,
        )
        second = TextElement(
            x_mm=12.0,
            y_mm=10.0,
            width_mm=50.0,
            height_mm=10.0,
            text="Second",
            font=font,
        )
        self.design.back.elements.extend([first, second])
        self.design.back.distribute_vertically(spacing_mm=5.0)
        print("Back element y positions:", [element.y_mm for element in self.design.back.elements])
        self.assertAlmostEqual(self.design.back.elements[0].y_mm, 0.0)
        self.assertAlmostEqual(self.design.back.elements[1].y_mm, 15.0)

    def test_palette_refresh_applies_backgrounds(self) -> None:
        new_palette = CardPalette(
            background=ColorStyle(red=255, green=255, blue=255),
            accent=ColorStyle(red=255, green=215, blue=64),
            text=ColorStyle(red=34, green=34, blue=34),
        )
        self.design.palette = new_palette
        self.design.refresh_palette()
        print("Refreshed backgrounds:", self.design.front.background, self.design.back.background)
        self.assertIs(self.design.front.background, new_palette.background)
        self.assertIs(self.design.back.background, new_palette.accent)

    def test_summary_reports_element_counts(self) -> None:
        font = CardFont(family="Inter", size_pt=12.0, weight="regular")
        self.design.front.elements.append(
            TextElement(
                x_mm=10.0,
                y_mm=12.0,
                width_mm=60.0,
                height_mm=18.0,
                text="Question",
                font=font,
            )
        )
        self.design.back.elements.append(
            ImageElement(
                x_mm=14.0,
                y_mm=16.0,
                width_mm=40.0,
                height_mm=24.0,
                source_path="./assets/example.png",
                description="Diagram",
            )
        )
        summary = self.design.summary()
        print("Card summary output:\n", summary)
        self.assertIn("Front elements: 1", summary)
        self.assertIn("Back elements: 1", summary)


if __name__ == "__main__":
    unittest.main()

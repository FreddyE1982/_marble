"""Card design primitives for the CardReview application.

The module focuses on ergonomic, assignment-driven APIs that empower
CardReview users to assemble visually engaging flash cards without dealing
with low-level drawing logic.

Example:
    >>> from cardreview.design import (
    ...     CardDesign,
    ...     CardDimensions,
    ...     CardPalette,
    ...     ColorStyle,
    ...     CardFont,
    ...     TextElement,
    ... )
    >>> design = CardDesign(
    ...     dimensions=CardDimensions(width_mm=88.0, height_mm=63.0),
    ...     palette=CardPalette(
    ...         background=ColorStyle(red=250, green=250, blue=252),
    ...         accent=ColorStyle(red=24, green=144, blue=255),
    ...         text=ColorStyle(red=25, green=28, blue=33),
    ...     ),
    ... )
    >>> prompt_font = CardFont(family="Inter", size_pt=18.0, weight="semibold")
    >>> answer_font = CardFont(family="Inter", size_pt=14.0, weight="regular")
    >>> front_block = TextElement(
    ...     x_mm=12.0,
    ...     y_mm=18.0,
    ...     width_mm=64.0,
    ...     height_mm=24.0,
    ...     text="What is the powerhouse of the cell?",
    ...     font=prompt_font,
    ... )
    >>> back_block = TextElement(
    ...     x_mm=12.0,
    ...     y_mm=18.0,
    ...     width_mm=64.0,
    ...     height_mm=24.0,
    ...     text="The mitochondrion produces ATP via cellular respiration.",
    ...     font=answer_font,
    ... )
    >>> design.front.elements.append(front_block)
    >>> design.back.elements.append(back_block)
    >>> design.front.align_center()
    >>> design.back.align_center()
    >>> design.front.distribute_vertically(spacing_mm=4.0)
    >>> print(design.summary())
    Card size: 88.0mm × 63.0mm\nFront elements: 1\nBack elements: 1
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass(slots=True)
class CardDimensions:
    """Physical card dimensions measured in millimetres.

    The CardReview desktop editor stores dimensions in millimetres, so
    this helper mirrors that convention and validates that both axes are
    positive numbers.

    Example:
        >>> from cardreview.design import CardDimensions
        >>> CardDimensions(width_mm=88.0, height_mm=63.0)
        CardDimensions(width_mm=88.0, height_mm=63.0)
    """

    width_mm: float
    height_mm: float

    def __post_init__(self) -> None:
        if self.width_mm <= 0.0:
            raise ValueError("width_mm must be positive")
        if self.height_mm <= 0.0:
            raise ValueError("height_mm must be positive")


@dataclass(slots=True)
class ColorStyle:
    """RGB colour triple for consistent palette assignment.

    Example:
        >>> from cardreview.design import ColorStyle
        >>> ColorStyle(red=12, green=34, blue=200)
        ColorStyle(red=12, green=34, blue=200)
    """

    red: int
    green: int
    blue: int

    def __post_init__(self) -> None:
        for component, name in ((self.red, "red"), (self.green, "green"), (self.blue, "blue")):
            if not 0 <= component <= 255:
                raise ValueError(f"{name} must be within 0-255")


@dataclass(slots=True)
class CardPalette:
    """Shared palette that the card faces can pull from.

    Example:
        >>> from cardreview.design import CardPalette, ColorStyle
        >>> CardPalette(
        ...     background=ColorStyle(red=255, green=255, blue=255),
        ...     accent=ColorStyle(red=40, green=120, blue=220),
        ...     text=ColorStyle(red=34, green=34, blue=34),
        ... )
        CardPalette(background=ColorStyle(red=255, green=255, blue=255), accent=ColorStyle(red=40, green=120, blue=220), text=ColorStyle(red=34, green=34, blue=34))
    """

    background: ColorStyle
    accent: ColorStyle
    text: ColorStyle


@dataclass(slots=True)
class CardFont:
    """Typography information used by text blocks on the card face.

    Example:
        >>> from cardreview.design import CardFont
        >>> CardFont(family="Inter", size_pt=16.0, weight="medium")
        CardFont(family='Inter', size_pt=16.0, weight='medium')
    """

    family: str
    size_pt: float
    weight: str

    def __post_init__(self) -> None:
        if self.size_pt <= 0.0:
            raise ValueError("size_pt must be positive")
        if not self.family:
            raise ValueError("family must not be empty")
        if not self.weight:
            raise ValueError("weight must not be empty")


@dataclass(slots=True)
class CardElement:
    """Base class for card elements positioned on a face.

    Example:
        >>> from cardreview.design import CardElement
        >>> CardElement(x_mm=10.0, y_mm=12.0, width_mm=40.0, height_mm=12.0)
        CardElement(x_mm=10.0, y_mm=12.0, width_mm=40.0, height_mm=12.0)
    """

    x_mm: float
    y_mm: float
    width_mm: float
    height_mm: float

    def __post_init__(self) -> None:
        if self.width_mm <= 0.0:
            raise ValueError("width_mm must be positive")
        if self.height_mm <= 0.0:
            raise ValueError("height_mm must be positive")

    def move_to(self, x_mm: float, y_mm: float) -> None:
        """Relocate the element by changing its anchor position.

        Example:
            >>> from cardreview.design import CardElement
            >>> element = CardElement(x_mm=10.0, y_mm=12.0, width_mm=40.0, height_mm=12.0)
            >>> element.move_to(24.0, 30.0)
            >>> element.x_mm, element.y_mm
            (24.0, 30.0)
        """

        self.x_mm = x_mm
        self.y_mm = y_mm

    def align_center(self, dimensions: CardDimensions) -> None:
        """Horizontally centre the element on the provided card dimensions.

        Example:
            >>> from cardreview.design import CardElement, CardDimensions
            >>> element = CardElement(x_mm=0.0, y_mm=12.0, width_mm=40.0, height_mm=12.0)
            >>> element.align_center(CardDimensions(width_mm=80.0, height_mm=60.0))
            >>> element.x_mm
            20.0
        """

        horizontal_margin = (dimensions.width_mm - self.width_mm) / 2.0
        self.x_mm = horizontal_margin


@dataclass(slots=True)
class TextElement(CardElement):
    """Text block rendered on a card face.

    Example:
        >>> from cardreview.design import CardFont, TextElement
        >>> font = CardFont(family="Inter", size_pt=14.0, weight="regular")
        >>> TextElement(
        ...     x_mm=10.0,
        ...     y_mm=15.0,
        ...     width_mm=60.0,
        ...     height_mm=20.0,
        ...     text="Hello",
        ...     font=font,
        ... )
        TextElement(x_mm=10.0, y_mm=15.0, width_mm=60.0, height_mm=20.0, text='Hello', font=CardFont(family='Inter', size_pt=14.0, weight='regular'), alignment='left')
    """

    text: str
    font: CardFont
    alignment: str = "left"

    def __post_init__(self) -> None:
        CardElement.__post_init__(self)
        if not self.text:
            raise ValueError("text must not be empty")
        if self.alignment not in {"left", "center", "right", "justify"}:
            raise ValueError("alignment must be one of left, center, right, justify")

    def change_alignment(self, alignment: str) -> None:
        """Update the text alignment preference.

        Example:
            >>> from cardreview.design import CardFont, TextElement
            >>> element = TextElement(
            ...     x_mm=10.0,
            ...     y_mm=15.0,
            ...     width_mm=60.0,
            ...     height_mm=20.0,
            ...     text="Hello",
            ...     font=CardFont(family="Inter", size_pt=14.0, weight="regular"),
            ... )
            >>> element.change_alignment("center")
            >>> element.alignment
            'center'
        """

        if alignment not in {"left", "center", "right", "justify"}:
            raise ValueError("alignment must be one of left, center, right, justify")
        self.alignment = alignment


@dataclass(slots=True)
class ImageElement(CardElement):
    """Image block placeholder on a card face.

    Example:
        >>> from cardreview.design import ImageElement
        >>> ImageElement(
        ...     x_mm=12.0,
        ...     y_mm=24.0,
        ...     width_mm=40.0,
        ...     height_mm=30.0,
        ...     source_path="./assets/mitochondrion.png",
        ...     description="Micrograph of a mitochondrion",
        ... )
        ImageElement(x_mm=12.0, y_mm=24.0, width_mm=40.0, height_mm=30.0, source_path='./assets/mitochondrion.png', description='Micrograph of a mitochondrion')
    """

    source_path: str
    description: str

    def __post_init__(self) -> None:
        CardElement.__post_init__(self)
        if not self.source_path:
            raise ValueError("source_path must not be empty")
        if not self.description:
            raise ValueError("description must not be empty")


@dataclass(slots=True)
class CardFace:
    """Single face of a card that aggregates card elements.

    Example:
        >>> from cardreview.design import CardFace, ColorStyle
        >>> face = CardFace(name="Front", background=ColorStyle(red=255, green=255, blue=255))
        >>> face
        CardFace(name='Front', background=ColorStyle(red=255, green=255, blue=255), elements=[])
    """

    name: str
    background: ColorStyle
    elements: List[CardElement] = field(default_factory=list)

    def align_center(self) -> None:
        """Horizontally centre every element using the widest element as reference.

        The routine keeps vertical positions intact and aligns all items based on
        the maximum width encountered so that columnar layouts become effortless.

        Example:
            >>> from cardreview.design import CardFace, CardDimensions, CardFont, TextElement, ColorStyle
            >>> face = CardFace(name="Front", background=ColorStyle(red=255, green=255, blue=255))
            >>> font = CardFont(family="Inter", size_pt=14.0, weight="regular")
            >>> face.elements.append(TextElement(x_mm=0.0, y_mm=10.0, width_mm=40.0, height_mm=12.0, text="Question", font=font))
            >>> face.elements.append(TextElement(x_mm=0.0, y_mm=28.0, width_mm=52.0, height_mm=12.0, text="Detail", font=font))
            >>> face.align_center()
            >>> [element.x_mm for element in face.elements]
            [6.0, 0.0]
        """

        if not self.elements:
            return
        max_width = max(element.width_mm for element in self.elements)
        max_height = max(element.height_mm for element in self.elements)
        dimensions = CardDimensions(width_mm=max_width, height_mm=max_height)
        for element in self.elements:
            element.align_center(dimensions)

    def distribute_vertically(self, spacing_mm: float) -> None:
        """Distribute elements vertically using the provided spacing.

        Elements are sorted by their current ``y_mm`` position before the
        operation; afterwards they are stacked with consistent spacing, making
        it simple to maintain tidy prompts and answer sections.

        Example:
            >>> from cardreview.design import CardFace, CardFont, TextElement, ColorStyle
            >>> face = CardFace(name="Back", background=ColorStyle(red=245, green=248, blue=255))
            >>> font = CardFont(family="Inter", size_pt=14.0, weight="regular")
            >>> face.elements.append(TextElement(x_mm=10.0, y_mm=20.0, width_mm=50.0, height_mm=10.0, text="Step 1", font=font))
            >>> face.elements.append(TextElement(x_mm=12.0, y_mm=30.0, width_mm=50.0, height_mm=10.0, text="Step 2", font=font))
            >>> face.distribute_vertically(spacing_mm=5.0)
            >>> [element.y_mm for element in face.elements]
            [0.0, 15.0]
        """

        if spacing_mm < 0.0:
            raise ValueError("spacing_mm must not be negative")
        if not self.elements:
            return
        def _y_position(element: CardElement) -> float:
            return element.y_mm

        ordered = sorted(self.elements, key=_y_position)
        self.elements[:] = ordered
        current_y = 0.0
        for element in self.elements:
            element.move_to(element.x_mm, current_y)
            current_y += element.height_mm + spacing_mm

    def apply_palette(self, palette: CardPalette, *, for_back_face: bool = False) -> None:
        """Apply the shared palette to the face.

        When ``for_back_face`` is true the accent colour is used as background
        to create contrast between front and back faces; otherwise the primary
        background colour is used.

        Example:
            >>> from cardreview.design import CardFace, CardPalette, ColorStyle
            >>> palette = CardPalette(
            ...     background=ColorStyle(red=255, green=255, blue=255),
            ...     accent=ColorStyle(red=12, green=100, blue=240),
            ...     text=ColorStyle(red=30, green=30, blue=30),
            ... )
            >>> face = CardFace(name="Front", background=palette.background)
            >>> face.apply_palette(palette)
            >>> face.background
            ColorStyle(red=255, green=255, blue=255)
        """

        if for_back_face:
            self.background = palette.accent
        else:
            self.background = palette.background


@dataclass(slots=True)
class CardDesign:
    """Complete card design that ties the faces and palette together.

    Example:
        >>> from cardreview.design import CardDesign, CardDimensions, CardPalette, ColorStyle
        >>> design = CardDesign(
        ...     dimensions=CardDimensions(width_mm=88.0, height_mm=63.0),
        ...     palette=CardPalette(
        ...         background=ColorStyle(red=255, green=255, blue=255),
        ...         accent=ColorStyle(red=10, green=132, blue=255),
        ...         text=ColorStyle(red=30, green=30, blue=30),
        ...     ),
        ... )
        >>> design.front.name
        'Front'
    """

    dimensions: CardDimensions
    palette: CardPalette
    front: CardFace = field(init=False)
    back: CardFace = field(init=False)

    def __post_init__(self) -> None:
        self.front = CardFace(name="Front", background=self.palette.background)
        self.back = CardFace(name="Back", background=self.palette.accent)

    def refresh_palette(self) -> None:
        """Re-apply the shared palette to both faces.

        Example:
            >>> from cardreview.design import CardDesign, CardDimensions, CardPalette, ColorStyle
            >>> design = CardDesign(
            ...     dimensions=CardDimensions(width_mm=88.0, height_mm=63.0),
            ...     palette=CardPalette(
            ...         background=ColorStyle(red=255, green=255, blue=255),
            ...         accent=ColorStyle(red=40, green=130, blue=240),
            ...         text=ColorStyle(red=20, green=20, blue=20),
            ...     ),
            ... )
            >>> design.palette = CardPalette(
            ...     background=ColorStyle(red=240, green=242, blue=245),
            ...     accent=ColorStyle(red=30, green=102, blue=220),
            ...     text=ColorStyle(red=30, green=30, blue=30),
            ... )
            >>> design.refresh_palette()
            >>> design.front.background
            ColorStyle(red=240, green=242, blue=245)
        """

        self.front.apply_palette(self.palette)
        self.back.apply_palette(self.palette, for_back_face=True)

    def summary(self) -> str:
        """Generate a textual summary describing the card design.

        Example:
            >>> from cardreview.design import CardDesign, CardDimensions, CardPalette, ColorStyle
            >>> design = CardDesign(
            ...     dimensions=CardDimensions(width_mm=88.0, height_mm=63.0),
            ...     palette=CardPalette(
            ...         background=ColorStyle(red=255, green=255, blue=255),
            ...         accent=ColorStyle(red=40, green=120, blue=255),
            ...         text=ColorStyle(red=30, green=30, blue=30),
            ...     ),
            ... )
            >>> design.summary()
            'Card size: 88.0mm × 63.0mm\nFront elements: 0\nBack elements: 0'
        """

        return (
            f"Card size: {self.dimensions.width_mm}mm × {self.dimensions.height_mm}mm\n"
            f"Front elements: {len(self.front.elements)}\n"
            f"Back elements: {len(self.back.elements)}"
        )

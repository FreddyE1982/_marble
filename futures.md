[Vorschlag]
Introduce reusable typography presets so CardReview designers can switch between saved font themes without recreating text blocks.
TODOS:
[TODO] Add a CardTypographyPreset class that stores font families and sizes for headings, body text, and annotations.
[TODO] Extend CardFace to apply a typography preset across existing TextElement instances.
[TODO] Update unit tests to validate preset application and ensure alignment helpers respect updated font metrics.

[Vorschlag]
Add printable bleed and safe-zone guides that automatically adjust element margins for professional print workflows.
TODOS:
[TODO] Implement a CardPrintGuide helper storing bleed and safe-zone offsets relative to CardDimensions.
[TODO] Enhance CardFace.align_center and distribute_vertically to honour the active CardPrintGuide margins.
[TODO] Create regression tests that confirm elements stay within the safe-zone after guide adjustments.

[Vorschlag]
Provide palette contrast analysis that warns designers about insufficient text/background contrast according to WCAG standards.
TODOS:
[TODO] Implement a contrast ratio calculator that compares ColorStyle combinations without relying on external libraries.
[TODO] Add a validation method on CardDesign that records warnings when palette contrast falls below configurable thresholds.
[TODO] Extend tests to cover both compliant and non-compliant palette configurations and verify warning messages.

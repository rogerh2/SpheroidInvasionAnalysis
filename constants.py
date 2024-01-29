import matplotlib.font_manager

MASKED = 'masked'
UNMASKED = 'unmasked'
TIME_UNIT = 'day'
PATTERN = rf'{TIME_UNIT}(\d+)'
FONT_SPEC = {'fontname': 'Arial', 'size': 22}

FONTS = list(set([f.name for f in matplotlib.font_manager.fontManager.ttflist]))
FONTS.sort()
ARIAL_IND = FONTS.index("Arial")

FONT_SIZES = [str(i) for i in range(1, 129)]
TICK_SIZES = [str(i) for i in range(6, 21)]

EXAMPLE_MOMENTS = [10867366442.0, 4085208145.0, 6782158297.0]

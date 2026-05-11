import os
font_dir = os.path.join(os.environ.get('WINDIR', 'C:/Windows'), 'Fonts')
candidates = ['msyh.ttc', 'msyhbd.ttc', 'simsun.ttc', 'simhei.ttf', 'simkai.ttf', 'STKAITI.TTF']
for f in candidates:
    path = os.path.join(font_dir, f)
    print(f"{f}: {os.path.exists(path)} -> {path}")

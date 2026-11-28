# -*- coding: utf-8 -*-
"""`--cli --help` 必须在中文 Windows（GBK 控制台）下也能正常打印。

历史回归：PARAMS 的说明文本里混进过 U+2212（真减号），argparse 的 print_help
把它写进 GBK 流时抛 UnicodeEncodeError，用户敲 `--help` 直接崩。
这里用 GBK 写死的 stdout（PYTHONIOENCODING=gbk）跑一遍子进程，断言：
  * 返回码 0
  * 输出里含有 `--no-trail-drop`（默认开启的开关必须有否定形式）
直接跑： python tests/test_cli_help.py
"""
import os
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _run_help(io_enc):
    env = dict(os.environ)
    env['PYTHONPATH'] = os.pathsep.join([str(ROOT / 'pydeps'), str(ROOT),
                                         str(ROOT / 'core')])
    env['PYTHONIOENCODING'] = io_enc
    env.pop('PYTHONUTF8', None)
    return subprocess.run([sys.executable, 'main.py', '--cli', '--help'],
                          cwd=str(ROOT), env=env, capture_output=True)


class CliHelp(unittest.TestCase):
    def test_help_does_not_crash_on_gbk_console(self):
        r = _run_help('gbk')
        self.assertEqual(r.returncode, 0,
                         '--help 在 GBK 控制台下崩了：\n'
                         + r.stderr.decode('utf-8', 'replace')[-2000:])
        text = (r.stdout + r.stderr).decode('utf-8', 'replace')
        self.assertIn('--no-trail-drop', text)

    def test_no_positive_unicode_minus_left(self):
        # 说明文本里残留 U+2212 会再次把 GBK 控制台搞崩，这里当守卫
        bad = []
        for p in sorted(list((ROOT / 'core').glob('*.py')) + [ROOT / 'main.py']):
            src = p.read_text(encoding='utf-8')
            for i, line in enumerate(src.splitlines(), 1):
                if '\u2212' in line:
                    bad.append('%s:%d' % (p.relative_to(ROOT), i))
        self.assertEqual(bad, [], '仍有 U+2212：%r' % bad[:10])


if __name__ == '__main__':
    unittest.main(verbosity=2)

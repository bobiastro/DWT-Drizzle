# -*- coding: utf-8 -*-
"""把 tests/ 下的脚本式自检挂到 unittest 上，让 `python -m unittest discover -s tests`
也能一次跑全。

仓库的测试一直是"跑脚本"风格（``python tests/test_x.py``，通过就 exit 0，判定靠
里面的 assert），没有 TestCase 类，所以 unittest 的 discover 收不到任何用例。
这里用一条用例把它们逐个在子进程里跑一遍：谁的退出码非 0 就带着它的输出失败。
"""
import pathlib
import subprocess
import sys
import unittest

HERE = pathlib.Path(__file__).resolve().parent
SELF = pathlib.Path(__file__).name


class ScriptSelfChecks(unittest.TestCase):
    def test_all_scripts(self):
        scripts = sorted(p for p in HERE.glob('test_*.py') if p.name != SELF)
        self.assertTrue(scripts, 'tests/ 下没找到任何 test_*.py')
        bad = []
        for p in scripts:
            r = subprocess.run([sys.executable, str(p)], cwd=str(HERE.parent),
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            if r.returncode != 0:
                bad.append('%s (exit %d)\n%s' % (
                    p.name, r.returncode,
                    r.stdout.decode('utf-8', 'replace')[-1500:]))
        self.assertEqual(bad, [], '自检失败:\n\n' + '\n\n'.join(bad))


if __name__ == '__main__':
    unittest.main()

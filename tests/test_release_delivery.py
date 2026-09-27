from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tarfile
import tempfile
import unittest

from scripts.build_demo_release import build_release
from scripts.release_support import check_models, source_files, verify_snapshot


class ReleaseDeliveryTests(unittest.TestCase):
    """交付包必须完整、自包含，同时排除用户数据并拒绝被替换的模型。"""

    def fixture(self, root: Path) -> None:
        """使用极小的假项目验证归档合同，不依赖真实模型或用户数据库。"""
        for directory in ('release', 'api', 'data', 'artifacts'):
            (root / directory).mkdir(parents=True)
        (root / 'api/main.py').write_text('# 中文示例源码\n', encoding='utf-8')
        (root / '.env').write_text('PRIVATE_VALUE=do-not-package', encoding='utf-8')
        (root / 'data/optiagent.sqlite3').write_bytes(b'private database')
        (root / 'artifacts/model.pt').write_bytes(b'fixed checkpoint')
        config = {'release_id': 'test-demo', 'source_globs': ['api/**/*.py'],
                  'files': ['release/demo.json'], 'checkpoints': {'learned': {
                      'path': 'artifacts/model.pt', 'bytes': 16,
                      'sha256': hashlib.sha256(b'fixed checkpoint').hexdigest()}}}
        (root / 'release/demo.json').write_text(json.dumps(config), encoding='utf-8')

    def test_archive_round_trip_excludes_private_files_and_checks_tampering(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / 'source'
            self.fixture(root)
            built = build_release(Path(directory) / 'delivery', root=root)
            with tarfile.open(built['archive']) as archive:
                names = archive.getnames()
                self.assertFalse(any('.env' in name or '.sqlite3' in name for name in names))
                archive.extractall(Path(directory) / 'extracted', filter='data')
            extracted = Path(directory) / 'extracted/OptiAgent-test-demo'
            self.assertEqual('frozen_release', verify_snapshot(extracted)['mode'])
            self.assertEqual(1, len(check_models(extracted)))
            rebuilt = build_release(Path(directory) / 'rebuilt', root=extracted)
            self.assertEqual(built['sha256'], rebuilt['sha256'])
            (extracted / 'api/main.py').write_text('# 被修改的代码\n', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, '快照不一致'):
                verify_snapshot(extracted)

    def test_same_content_builds_identical_archive_and_existing_output_is_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / 'source'
            self.fixture(root)
            first = build_release(Path(directory) / 'first', root=root)
            second = build_release(Path(directory) / 'second', root=root)
            self.assertEqual(first['sha256'], second['sha256'])
            original = Path(first['archive']).read_bytes()
            with self.assertRaises(FileExistsError):
                build_release(Path(directory) / 'first', root=root)
            self.assertEqual(original, Path(first['archive']).read_bytes())

    def test_corrupted_checkpoint_cannot_enter_full_release(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / 'source'
            self.fixture(root)
            (root / 'artifacts/model.pt').write_bytes(b'changed checkpoint')
            with self.assertRaisesRegex(ValueError, '检查点与交付清单不符'):
                build_release(Path(directory) / 'delivery', root=root)
            self.assertFalse((Path(directory) / 'delivery').exists())

    def test_core_archive_does_not_require_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / 'source'
            self.fixture(root)
            (root / 'artifacts/model.pt').unlink()
            built = build_release(Path(directory) / 'delivery', profile='core', root=root)
            self.assertEqual(0, built['models'])

    def test_source_symlink_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / 'source'
            self.fixture(root)
            (root / 'api/leak.py').symlink_to(root / 'data/optiagent.sqlite3')
            with self.assertRaisesRegex(ValueError, '符号链接'):
                source_files(root)


if __name__ == '__main__':
    unittest.main()

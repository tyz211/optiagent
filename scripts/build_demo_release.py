from __future__ import annotations

import argparse
import gzip
import io
import json
from pathlib import Path
import subprocess
import tarfile
import tempfile
import shutil
import sys

# 同时支持命令行直接执行与回归测试中的模块导入。
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.release_support import ROOT, check_models, configuration, digest, inventory, inventory_digest, source_files


def build_release(output: Path, profile: str = 'full', root: Path = ROOT) -> dict:
    """生成独立且不可覆盖的交付包；既有未提交代码按实际内容进入快照。"""
    config = configuration(root)
    sources = source_files(root)
    models = []
    if profile == 'full':
        check_models(root)
        models = [item['path'] for item in config['checkpoints'].values()]
    output.mkdir(parents=True, exist_ok=False)
    original_manifest = root / 'release-manifest.json'
    if original_manifest.exists():
        # 解压包没有 Git 历史，重打包时保留原始来源元数据。
        base_commit = json.loads(original_manifest.read_text(encoding='utf-8')).get('base_commit')
    else:
        try:
            base_commit = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=root,
                                         text=True, capture_output=True, check=True).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            base_commit = None
    # 临时拷贝后再计算摘要，确保归档内容与清单严格对应。
    with tempfile.TemporaryDirectory(prefix='optiagent-package-') as temporary:
        staging = Path(temporary)
        for name in sources + models:
            target = staging / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(root / name, target)
        if profile == 'full':
            check_models(staging)
        files = inventory(staging, sources)
        manifest = {'schema_version': '1.0', 'release_id': config['release_id'],
                    'profile': profile, 'base_commit': base_commit,
                    'source_sha256': inventory_digest(files), 'source_files': files,
                    'model_files': inventory(staging, models),
                    'scope': '本地 Demo 源码和可选固定权重；不包含环境、许可证、用户数据或完整研究数据集。'}
        manifest_text = json.dumps(manifest, ensure_ascii=False, indent=2) + '\n'
        (staging / 'release-manifest.json').write_text(manifest_text, encoding='utf-8')
        archive = output / f"OptiAgent-{config['release_id']}-{profile}.tar.gz"
        prefix = f"OptiAgent-{config['release_id']}"
        # 固定压缩头、时间与权限，相同内容可重复生成相同压缩包。
        with archive.open('xb') as raw, gzip.GzipFile(filename='', mode='wb', fileobj=raw, mtime=0) as zipped:
            with tarfile.open(fileobj=zipped, mode='w') as tar:
                for name in sorted(sources + models + ['release-manifest.json']):
                    content = (staging / name).read_bytes()
                    info = tarfile.TarInfo(f'{prefix}/{name}')
                    info.size = len(content)
                    info.mode = 0o755 if name == 'start.sh' else 0o644
                    tar.addfile(info, io.BytesIO(content))
    (output / 'release-manifest.json').write_text(manifest_text, encoding='utf-8')
    (output / 'SHA256SUMS').write_text(f'{digest(archive)}  {archive.name}\n', encoding='utf-8')
    return {'archive': str(archive), 'sha256': digest(archive),
            'source_sha256': manifest['source_sha256'], 'source_files': len(sources), 'models': len(models)}


def main() -> None:
    """打包只读取项目，输出目录必须为新目录；不自动提交或发布。"""
    parser = argparse.ArgumentParser(description='生成可复核的 OptiAgent 本地交付包')
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--profile', choices=('core', 'full'), default='full')
    args = parser.parse_args()
    print(json.dumps(build_release(args.output.resolve(), args.profile), ensure_ascii=False))


if __name__ == '__main__':
    main()

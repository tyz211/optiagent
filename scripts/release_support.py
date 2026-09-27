from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def configuration(root: Path = ROOT) -> dict:
    """交付范围由版本化清单明确声明，避免把用户数据一起打包。"""
    return json.loads((root / 'release/demo.json').read_text(encoding='utf-8'))


def digest(path: Path) -> str:
    """流式计算摘要，兼容未来更大的模型文件。"""
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def safe_path(root: Path, relative: str) -> Path:
    """拒绝越界路径与符号链接，保证交付物来自指定目录。"""
    path = root / relative
    if Path(relative).is_absolute() or '..' in Path(relative).parts:
        raise ValueError(f'交付路径越界：{relative}')
    # 只检查项目内部，系统临时目录本身可能是平台提供的链接。
    for index in range(1, len(Path(relative).parts) + 1):
        if root.joinpath(*Path(relative).parts[:index]).is_symlink():
            raise ValueError(f'交付文件不能是符号链接：{relative}')
    if not path.is_file() or not path.resolve().is_relative_to(root.resolve()):
        raise ValueError(f'交付文件不存在或越界：{relative}')
    return path


def source_files(root: Path = ROOT) -> list[str]:
    """仅收集源码、公开说明与明确列出的示例，不读取数据库和本地配置。"""
    config = configuration(root)
    paths = set(config['files'])
    for pattern in config['source_globs']:
        paths.update(path.relative_to(root).as_posix() for path in root.glob(pattern)
                     if '__pycache__' not in path.parts)
    for relative in paths:
        safe_path(root, relative)
    return sorted(paths)


def inventory(root: Path, paths: list[str]) -> dict:
    """逐文件内容摘要是工作区快照的身份，不依赖未提交代码的 Git 编号。"""
    return {name: {'sha256': digest(safe_path(root, name)),
                   'bytes': safe_path(root, name).stat().st_size} for name in sorted(paths)}


def inventory_digest(files: dict) -> str:
    """使用稳定序列化，使相同源码始终得到相同版本摘要。"""
    return hashlib.sha256(json.dumps(files, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def check_models(root: Path = ROOT) -> dict:
    """加载权重前核对固定研究产物的摘要，损坏或替换必须明确报错。"""
    results = {}
    for name, record in configuration(root)['checkpoints'].items():
        path = safe_path(root, record['path'])
        if digest(path) != record['sha256'] or path.stat().st_size != record['bytes']:
            raise ValueError(f'{name} 检查点与交付清单不符。')
        results[name] = record['sha256']
    return results


def verify_snapshot(root: Path = ROOT) -> dict:
    """源码目录返回当前身份；交付目录还必须通过冻结清单核验。"""
    current = inventory(root, source_files(root))
    manifest_path = root / 'release-manifest.json'
    if not manifest_path.exists():
        return {'mode': 'working_tree', 'source_sha256': inventory_digest(current)}
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if manifest['source_files'] != current:
        raise ValueError('源码与交付快照不一致，请重新解压原始交付包。')
    if manifest['source_sha256'] != inventory_digest(current):
        raise ValueError('源码摘要与交付清单不一致。')
    for name, record in manifest['model_files'].items():
        if inventory(root, [name])[name] != record:
            raise ValueError(f'交付模型摘要不一致：{name}')
    return {'mode': 'frozen_release', 'source_sha256': manifest['source_sha256'],
            'release_id': manifest['release_id'], 'profile': manifest['profile']}


def locked_versions(root: Path, filename: str) -> dict[str, str]:
    """读取简单的固定版本清单，不执行文件内容或访问包索引。"""
    versions = {}
    for line in (root / filename).read_text(encoding='utf-8').splitlines():
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        if line.startswith('-r '):
            versions.update(locked_versions(root, line[3:]))
        else:
            name, version = line.split('==')
            versions[name] = version
    return versions

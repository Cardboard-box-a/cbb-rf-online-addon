from __future__ import annotations

import csv
import os
import re
import shutil
import struct
import tempfile
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import bpy
from bpy.props import BoolProperty, EnumProperty, IntProperty, StringProperty
from bpy.types import Context, Event, Operator


DEFAULT_CLIENT_FOLDER = "/mnt/hdd1/Dev/RF Dev/rf client"
DEFAULT_EXCEL_DATA_FOLDER = "/mnt/hdd1/Dev/RF Dev/RF_Online_Engine_WIP/RF Online Sourcecode/excel_data"

ARMOR_SLOT_BY_TABLE = {
    "00_FaceItem/FaceItem.txt": ("FACE", 1),
    "01_UpperItem/UpperItem.txt": ("UPPER", 2),
    "02_LowerItem/LowerItem.txt": ("LOWER", 3),
    "03_GauntletItem/GauntletItem.txt": ("GLOVES", 4),
    "04_ShoeItem/ShoeItem.txt": ("SHOES", 5),
    "05_HelmetItem/HelmetItem.txt": ("HELMET", 0),
}

UNIT_PART_TABLES = {
    "Head": "37_UnitHead/UnitHead.txt",
    "Upper": "38_UnitUpper/UnitUpper.txt",
    "Lower": "39_UnitLower/UnitLower.txt",
    "Arms": "40_UnitArms/UnitArms.txt",
    "Shoulder": "41_UnitShoulder/UnitShoulder.txt",
    "Back": "42_UnitBack/UnitBack.txt",
    "BullaetArms": "43_UnitBullet/UnitBullet.txt",
    "BulletShould": "43_UnitBullet/UnitBullet.txt",
}

RACE_PREFIX_TO_BONE = {
    "ACCRETIA": "Accretia.bn",
    "BELMALE": "BelMale.bn",
    "BELFEMALE": "BelFemale.bn",
    "CORMALE": "CorMale.bn",
    "CORFEMALE": "CorFemale.bn",
}

UNIT_BONE_BY_NAME = {
    "GOLIATH": "GOLIATH.bn",
    "CATAPULT": "CATAPULT.bn",
}

RESOURCE_SECTIONS = {
    "ANIMUS": "2_Animus",
    "MONSTER": "1_Monster",
    "NPC": "4_NPC",
    "GUARD_TOWER": "3_GuardTower",
}

EXPORT_MODES = (
    "PLAYER_PACKS",
    "UNIT_PACKS",
    "PLAYER_INDIVIDUAL",
    "ITEM_INDIVIDUAL",
    "UNIT_INDIVIDUAL",
    "ANIMUS",
    "MONSTER",
    "NPC",
    "GUARD_TOWER",
)

CLIENT_BONE_ROOTS = {
    "ANIMUS": ".\\CHARACTER\\ANIMUS\\BONE\\",
    "MONSTER": ".\\CHARACTER\\MONSTER\\BONE\\",
    "NPC": ".\\CHARACTER\\NPC\\BONE\\",
    "GUARD_TOWER": ".\\CHARACTER\\GUARDTOWER\\BONE\\",
}


@dataclass
class TableRow:
    source: Path
    fields: dict[str, str]


@dataclass
class ResourceEntry:
    id: int
    bone_id: int | None
    path_name: str
    file_name: str
    texture_path: str = ""


@dataclass
class ExportJob:
    name: str
    kind: str
    bone: str | None
    meshes: list[str] = field(default_factory=list)
    animations: list[str] = field(default_factory=list)


class RFSArchive:
    def __init__(self, path: Path):
        self.path = path
        self._entries: dict[str, tuple[int, int, str]] | None = None

    def entries(self) -> dict[str, tuple[int, int, str]]:
        if self._entries is not None:
            return self._entries

        entries: dict[str, tuple[int, int, str]] = {}
        with self.path.open("rb") as opened_file:
            file_count = struct.unpack("<I", opened_file.read(4))[0]
            for _ in range(file_count):
                raw_name = opened_file.read(56)
                file_name = raw_name.split(b"\x00")[0].decode("euc-kr")
                file_offset = struct.unpack("<I", opened_file.read(4))[0]
                file_size = struct.unpack("<I", opened_file.read(4))[0]
                entries[file_name.casefold()] = (file_offset, file_size, file_name)
        self._entries = entries
        return entries

    def has(self, file_name: str) -> bool:
        return file_name.casefold() in self.entries()

    def extract(self, file_name: str, output_path: Path) -> Path:
        file_offset, file_size, stored_name = self.entries()[file_name.casefold()]
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("rb") as opened_file:
            opened_file.seek(file_offset)
            data = opened_file.read(file_size)
        output_path.write_bytes(data)
        return output_path


class ClientAssetResolver:
    def __init__(self, client_folder: Path, temp_root: Path):
        self.client_folder = client_folder
        self.temp_root = temp_root
        self._rfs_by_dir: dict[Path, list[RFSArchive]] = {}

    @staticmethod
    def normalize_client_path(path_name: str) -> Path:
        normalized = path_name.replace("\\", "/").replace("./", "").replace(".\\", "")
        parts = [part for part in normalized.split("/") if part and part != "."]
        return Path(*parts)

    def resolve_loose(self, path_name: str, file_name: str) -> Path | None:
        directory = self.resolve_client_dir(self.normalize_client_path(path_name))
        if directory is None:
            return None
        candidate = self.find_file_icase(directory, file_name)
        if candidate:
            return candidate
        return None

    def resolve_for_import(self, path_name: str, file_name: str) -> Path | None:
        rel_dir = self.normalize_client_path(path_name)
        source_dir = self.resolve_client_dir(rel_dir)
        if source_dir is None:
            return None
        archive = self.find_archive_containing(source_dir, file_name)
        if archive is not None:
            temp_path = self.temp_root / rel_dir / file_name
            archive.extract(file_name, temp_path)
            self.mirror_sibling_rfs_for_textures(rel_dir)
            return temp_path

        loose = self.resolve_loose(path_name, file_name)
        if loose:
            return loose

        extracted = self.find_file_in_extract_dirs(source_dir, file_name)
        if extracted:
            return extracted

        return None

    def find_archive_containing(self, source_dir: Path, file_name: str) -> RFSArchive | None:
        for archive in self.archives_in(source_dir):
            if archive.has(file_name):
                return archive
        return None

    def mirror_sibling_rfs_for_textures(self, rel_dir: Path) -> None:
        parent = rel_dir.parent
        source_parent = self.resolve_client_dir(parent)
        if source_parent is None:
            return
        source_tex = self.find_child_dir_icase(source_parent, "Tex")
        if source_tex is None:
            source_tex = self.find_child_dir_icase(source_parent, "TEX")
        if source_tex is None:
            return

        temp_tex = self.temp_root / parent / source_tex.name
        temp_tex.mkdir(parents=True, exist_ok=True)
        for rfs_file in source_tex.iterdir():
            if rfs_file.is_file() and rfs_file.suffix.casefold() == ".rfs":
                target = temp_tex / rfs_file.name
                if not target.exists():
                    try:
                        target.symlink_to(rfs_file)
                    except OSError:
                        shutil.copy2(rfs_file, target)

    def find_file_in_extract_dirs(self, source_dir: Path, file_name: str) -> Path | None:
        if not source_dir.exists():
            return None
        for child in source_dir.iterdir():
            if child.is_dir() and child.name.casefold().endswith(".rfs.extract"):
                found = self.find_file_icase(child, file_name)
                if found:
                    return found
        return None

    def archives_in(self, source_dir: Path) -> list[RFSArchive]:
        if source_dir in self._rfs_by_dir:
            return self._rfs_by_dir[source_dir]
        archives: list[RFSArchive] = []
        if source_dir.exists():
            for child in source_dir.iterdir():
                if child.is_file() and child.suffix.casefold() == ".rfs":
                    archives.append(RFSArchive(child))
        self._rfs_by_dir[source_dir] = archives
        return archives

    def resolve_client_dir(self, rel_dir: Path) -> Path | None:
        current = self.client_folder
        if str(rel_dir) in ("", "."):
            return current
        for segment in rel_dir.parts:
            next_dir = self.find_child_dir_icase(current, segment)
            if next_dir is None:
                return None
            current = next_dir
        return current

    @staticmethod
    def find_child_dir_icase(parent: Path, target_name: str) -> Path | None:
        if not parent.exists():
            return None
        target = target_name.casefold()
        for child in parent.iterdir():
            if child.is_dir() and child.name.casefold() == target:
                return child
        return None

    @staticmethod
    def find_file_icase(directory: Path, file_name: str) -> Path | None:
        if not directory.exists():
            return None
        target = file_name.casefold()
        for child in directory.iterdir():
            if child.is_file() and child.name.casefold() == target:
                return child
        return None


def read_table(path: Path) -> list[TableRow]:
    last_error: UnicodeDecodeError | None = None
    for encoding in ("utf-8-sig", "cp949", "euc-kr"):
        try:
            with path.open("r", encoding=encoding, newline="") as opened_file:
                rows = list(csv.reader(opened_file, delimiter="\t"))
            break
        except UnicodeDecodeError as error:
            last_error = error
    else:
        with path.open("r", encoding="cp949", errors="replace", newline="") as opened_file:
            rows = list(csv.reader(opened_file, delimiter="\t"))
    if len(rows) < 2:
        return []
    headers = [header.strip() for header in rows[1]]
    result: list[TableRow] = []
    for values in rows[2:]:
        if not values or not values[0].strip():
            continue
        fields = {headers[i]: values[i].strip() for i in range(min(len(headers), len(values))) if headers[i]}
        result.append(TableRow(path, fields))
    return result


def parse_int(value: str, default: int = 0) -> int:
    if value is None or value == "":
        return default
    text = str(value).strip()
    try:
        return int(text, 16)
    except ValueError:
        return int(float(text))


def parse_decimal(value: str, default: int = 0) -> int:
    if value is None or value == "":
        return default
    try:
        return int(float(str(value).strip()))
    except ValueError:
        return default


def read_resource_entries(path: Path) -> list[ResourceEntry]:
    entries: list[ResourceEntry] = []
    for row in read_table(path):
        fields = row.fields
        entries.append(
            ResourceEntry(
                id=parse_int(fields.get("ID", "0")),
                bone_id=parse_int(fields["BoneID"]) if "BoneID" in fields else None,
                path_name=fields.get("PathName", ""),
                file_name=fields.get("FileName", ""),
                texture_path=fields.get("TexutrePath", fields.get("TexturePath", "")),
            )
        )
    return entries


def model_to_resource_id(model: str, slot_index: int) -> int | None:
    if not model:
        return None
    try:
        model_value = int(model, 16)
    except ValueError:
        return None
    race = (model_value >> 20) & 0xF
    variant = model_value & 0xFF
    return (race << 20) | (slot_index << 8) | variant


def model_to_unit_resource_id(model: str) -> int | None:
    if not model:
        return None
    try:
        return int(model, 16)
    except ValueError:
        return None


def player_race_prefix(file_name: str) -> str | None:
    upper = Path(file_name).stem.upper()
    for prefix in sorted(RACE_PREFIX_TO_BONE, key=len, reverse=True):
        if upper.startswith(prefix + "_"):
            return prefix
    return None


def unit_name_from_mesh(file_name: str) -> str | None:
    upper = Path(file_name).stem.upper()
    if upper.startswith("GOLIATH_"):
        return "GOLIATH"
    if upper.startswith("CATAPULT_"):
        return "CATAPULT"
    return None


def build_player_pack_jobs(excel_root: Path) -> list[ExportJob]:
    grouped: dict[tuple[str, str], dict[str, ResourceEntry]] = {}
    default_faces: dict[str, ResourceEntry] = {}
    default_helmets: dict[str, ResourceEntry] = {}
    armor_pattern = re.compile(
        r"^(ACCRETIA|BELFEMALE|BELMALE|CORFEMALE|CORMALE)_ARMOR_(HELMET|UPPER|LOWER|GLOVES|SHOES)_(\d+)$",
        re.IGNORECASE,
    )
    default_pattern = re.compile(
        r"^(ACCRETIA|BELFEMALE|BELMALE|CORFEMALE|CORMALE)_DEFAULT_(FACE|HELMET)_(\d+)$",
        re.IGNORECASE,
    )

    for entry in read_resource_entries(excel_root / "Resource.edf/0_Player/Mesh.txt"):
        stem = Path(entry.file_name).stem.upper()
        armor_match = armor_pattern.match(stem)
        if armor_match:
            race, slot_name, suffix = armor_match.groups()
            grouped.setdefault((race.upper(), suffix), {})[slot_name.upper()] = entry
            continue

        default_match = default_pattern.match(stem)
        if default_match:
            race, slot_name, suffix = default_match.groups()
            if suffix == "000" and slot_name.upper() == "FACE":
                default_faces[race.upper()] = entry
            elif suffix == "000" and slot_name.upper() == "HELMET":
                default_helmets[race.upper()] = entry

    jobs: list[ExportJob] = []
    required_slots = {"UPPER", "LOWER", "GLOVES", "SHOES"}
    ordered_slots = ["FACE", "HAIR", "UPPER", "LOWER", "GLOVES", "SHOES", "HELMET"]
    for (race, suffix), parts in sorted(grouped.items()):
        if not required_slots.issubset(parts):
            continue
        if race in default_faces:
            parts["FACE"] = default_faces[race]
        if race != "ACCRETIA" and race in default_helmets:
            parts["HAIR"] = default_helmets[race]
        elif "HELMET" not in parts and race in default_helmets:
            parts["HELMET"] = default_helmets[race]
        meshes = [parts[slot].path_name + parts[slot].file_name for slot in ordered_slots if slot in parts]
        jobs.append(
            ExportJob(
                name=f"PLAYER_{race}_SET_{suffix}",
                kind="player_pack",
                bone=f".\\CHARACTER\\PLAYER\\BONE\\{RACE_PREFIX_TO_BONE[race]}",
                meshes=meshes,
            )
        )
    return jobs


def build_unit_pack_jobs(excel_root: Path) -> list[ExportJob]:
    mesh_entries = {entry.id: entry for entry in read_resource_entries(excel_root / "Resource.edf/6_Unit/Mesh.txt")}
    grouped: dict[tuple[str, int], dict[str, ResourceEntry]] = {}
    rel_tables = sorted(set(UNIT_PART_TABLES.values()))
    for rel_table in rel_tables:
        slot_name = Path(rel_table).parent.name.split("_", 1)[-1].upper()
        for row in read_table(excel_root / "Item.edf" / rel_table):
            resource_id = model_to_unit_resource_id(row.fields.get("Model", ""))
            if resource_id is None or resource_id not in mesh_entries:
                continue
            entry = mesh_entries[resource_id]
            unit_name = unit_name_from_mesh(entry.file_name)
            if not unit_name:
                continue
            level = parse_decimal(row.fields.get("LvLim", "0"), 0)
            if level <= 0 or level >= 100:
                continue
            grouped.setdefault((unit_name, level), {})[f"{slot_name}_{Path(entry.file_name).stem.upper()}"] = entry

    jobs: list[ExportJob] = []
    for (unit_name, level), parts in sorted(grouped.items()):
        if len(parts) < 4:
            continue
        meshes = [entry.path_name + entry.file_name for entry in parts.values()]
        jobs.append(
            ExportJob(
                name=f"UNIT_{unit_name}_LV{level:03d}",
                kind="unit_pack",
                bone=f".\\CHARACTER\\UNIT\\BONE\\{UNIT_BONE_BY_NAME[unit_name]}",
                meshes=meshes,
            )
        )
    return jobs


def build_individual_mesh_jobs(excel_root: Path, resource_section: str, kind: str) -> list[ExportJob]:
    mesh_entries = read_resource_entries(excel_root / f"Resource.edf/{resource_section}/Mesh.txt")
    jobs: list[ExportJob] = []
    seen_names: set[str] = set()
    for entry in mesh_entries:
        name = Path(entry.file_name).stem.upper()
        if not entry.file_name or name in seen_names:
            continue
        seen_names.add(name)
        bone = None
        if kind == "player_individual":
            race = player_race_prefix(entry.file_name)
            if race:
                bone = f".\\CHARACTER\\PLAYER\\BONE\\{RACE_PREFIX_TO_BONE[race]}"
        elif kind == "unit_individual":
            unit_name = unit_name_from_mesh(entry.file_name)
            if unit_name:
                bone = f".\\CHARACTER\\UNIT\\BONE\\{UNIT_BONE_BY_NAME[unit_name]}"

        jobs.append(
            ExportJob(
                name=f"{kind.upper()}_{name}",
                kind=kind,
                bone=bone,
                meshes=[entry.path_name + entry.file_name],
            )
        )
    return jobs


def build_actor_jobs(excel_root: Path, actor_kind: str) -> list[ExportJob]:
    section = RESOURCE_SECTIONS[actor_kind]
    bone_entries = {entry.id: entry for entry in read_resource_entries(excel_root / f"Resource.edf/{section}/Bone.txt")}
    mesh_entries = read_resource_entries(excel_root / f"Resource.edf/{section}/Mesh.txt")
    ani_entries = read_resource_entries(excel_root / f"Resource.edf/{section}/Ani.txt")

    jobs: list[ExportJob] = []
    for mesh in mesh_entries:
        mesh_stem = Path(mesh.file_name).stem.upper()
        if not mesh_stem:
            continue

        bone_id = mesh.bone_id if mesh.bone_id is not None else mesh.id
        bone_entry = bone_entries.get(bone_id) or bone_entries.get(mesh.id)
        bone_ref = None
        if bone_entry:
            bone_ref = bone_entry.path_name + bone_entry.file_name
        else:
            bone_ref = CLIENT_BONE_ROOTS.get(actor_kind, "") + f"{mesh_stem}.BN"

        animation_refs = []
        for ani in ani_entries:
            ani_stem = Path(ani.file_name).stem.upper()
            if ani_stem.startswith(mesh_stem + "_"):
                animation_refs.append(ani.path_name + ani.file_name)

        jobs.append(
            ExportJob(
                name=f"{actor_kind}_{mesh_stem}",
                kind=actor_kind.lower(),
                bone=bone_ref,
                meshes=[mesh.path_name + mesh.file_name],
                animations=animation_refs,
            )
        )
    return jobs


def build_jobs_for_mode(excel_root: Path, mode: str) -> list[ExportJob]:
    if mode == "PLAYER_PACKS":
        return build_player_pack_jobs(excel_root)
    if mode == "UNIT_PACKS":
        return build_unit_pack_jobs(excel_root)
    if mode == "PLAYER_INDIVIDUAL":
        return build_individual_mesh_jobs(excel_root, "0_Player", "player_individual")
    if mode == "ITEM_INDIVIDUAL":
        return build_individual_mesh_jobs(excel_root, "5_Item", "item_individual")
    if mode == "UNIT_INDIVIDUAL":
        return build_individual_mesh_jobs(excel_root, "6_Unit", "unit_individual")
    return build_actor_jobs(excel_root, mode)


def split_client_ref(client_ref: str) -> tuple[str, str]:
    normalized = client_ref.replace("\\", "/").replace("./", "").replace(".\\", "")
    return str(Path(normalized).parent) + "\\", Path(normalized).name


def clear_scene() -> None:
    try:
        bpy.ops.object.mode_set(mode="OBJECT")
    except RuntimeError:
        pass

    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()

    for action in list(bpy.data.actions):
        action.use_fake_user = False
        bpy.data.actions.remove(action, do_unlink=True)

    for collection in list(bpy.data.collections):
        bpy.data.collections.remove(collection, do_unlink=True)

    for datablock_collection in (bpy.data.meshes, bpy.data.armatures, bpy.data.materials, bpy.data.images):
        for datablock in list(datablock_collection):
            if datablock.users == 0:
                datablock_collection.remove(datablock)


def call_file_operator(operator_id: str, filepath: Path, **kwargs) -> set[str]:
    directory = str(filepath.parent) + os.sep
    files = [{"name": filepath.name}]
    return getattr(getattr(bpy.ops, operator_id.split(".")[0]), operator_id.split(".")[1])(
        directory=directory,
        filepath=str(filepath),
        files=files,
        **kwargs,
    )


def select_exportable_objects() -> None:
    bpy.ops.object.select_all(action="DESELECT")
    for obj in bpy.context.scene.objects:
        if obj.type in {"ARMATURE", "MESH", "EMPTY"} and "Bone Shapes" not in [collection.name for collection in obj.users_collection]:
            obj.select_set(True)


def export_job(job: ExportJob, resolver: ClientAssetResolver, output_dir: Path, include_animations: bool, log=None) -> tuple[bool, str]:
    clear_scene()
    output_dir.mkdir(parents=True, exist_ok=True)
    warnings: list[str] = []

    if job.bone:
        bone_path_name, bone_file = split_client_ref(job.bone)
        bone_path = resolver.resolve_loose(bone_path_name, bone_file)
        if bone_path is None:
            warnings.append(f"missing bone: {job.bone}; continuing mesh-only")
            if log:
                log.write(f"  warning: missing bone: {job.bone}; continuing mesh-only\n")
        else:
            if log:
                log.write(f"  bone: {job.bone} -> {bone_path}\n")
            call_file_operator("cbb.bn_skeleton_import", bone_path)

    for mesh_ref in job.meshes:
        mesh_path_name, mesh_file = split_client_ref(mesh_ref)
        mesh_path = resolver.resolve_for_import(mesh_path_name, mesh_file)
        loose_fallback = resolver.resolve_loose(mesh_path_name, mesh_file)
        if mesh_path is None and loose_fallback is not None:
            mesh_path = loose_fallback
        if mesh_path is None:
            return False, f"Missing mesh: {mesh_ref}"
        if log:
            log.write(f"  mesh: {mesh_ref} -> {mesh_path}\n")
        select_exportable_objects()
        call_file_operator(
            "cbb.msh_import",
            mesh_path,
            apply_to_armature_in_selected=True,
            preserve_parenting_relationships=True,
            assign_bind_matrices_to_objects=False,
        )

    if include_animations:
        select_exportable_objects()
        for ani_ref in job.animations:
            ani_path_name, ani_file = split_client_ref(ani_ref)
            ani_path = resolver.resolve_for_import(ani_path_name, ani_file)
            if ani_path is None:
                warnings.append(f"missing animation: {ani_ref}")
                if log:
                    log.write(f"  warning: missing animation: {ani_ref}\n")
                continue
            if log:
                log.write(f"  ani: {ani_ref} -> {ani_path}\n")
            call_file_operator("cbb.ani_import", ani_path, apply_to_selected_objects=True, ignore_not_found=True)

    try:
        select_exportable_objects()
        glb_path = output_dir / f"{job.name}.glb"
        if log:
            log.write(f"  export: {glb_path}\n")
        bpy.ops.export_scene.gltf(
            filepath=str(glb_path),
            export_format="GLB",
            use_selection=True,
            export_animations=include_animations,
            export_nla_strips=True,
        )
        if warnings:
            return True, f"exported with {len(warnings)} warning(s)"
        return True, "exported"
    finally:
        clear_scene()


class CBB_OT_BatchExportGLB(Operator):
    bl_idname = "cbb.batch_glb_export"
    bl_label = "Batch Export RF glB"
    bl_options = {"PRESET"}

    client_folder: StringProperty(name="Client Folder", subtype="DIR_PATH", default=DEFAULT_CLIENT_FOLDER)  # type: ignore
    excel_data_folder: StringProperty(name="Excel Data Folder", subtype="DIR_PATH", default=DEFAULT_EXCEL_DATA_FOLDER)  # type: ignore
    output_folder: StringProperty(name="Output Folder", subtype="DIR_PATH", default="//rf_glb_exports")  # type: ignore
    mode: EnumProperty(
        name="Mode",
        items=[
            ("ALL", "Export All", "Export every batch mode in one run"),
            ("PLAYER_PACKS", "Player equipment packs", "Export complete five-part player armor packs"),
            ("UNIT_PACKS", "MAU packs", "Export coherent MAU frame/level packs"),
            ("PLAYER_INDIVIDUAL", "Player individual meshes", "Export each player mesh as its own glB"),
            ("ITEM_INDIVIDUAL", "Item individual meshes", "Export each item mesh as its own glB"),
            ("UNIT_INDIVIDUAL", "MAU individual meshes", "Export each MAU mesh as its own glB"),
            ("ANIMUS", "Animus grouped", "Export animus skeleton, mesh, and optional animations"),
            ("MONSTER", "Monster grouped", "Export monster skeleton, mesh, and optional animations"),
            ("NPC", "NPC grouped", "Export NPC skeleton, mesh, and optional animations"),
            ("GUARD_TOWER", "Guard tower grouped", "Export guard tower skeleton, mesh, and optional animations"),
        ],
        default="PLAYER_PACKS",
    )  # type: ignore
    limit: IntProperty(name="Limit", default=0, min=0, description="0 exports all jobs")  # type: ignore
    dry_run: BoolProperty(name="Dry Run", default=True, description="Print jobs without exporting")  # type: ignore
    include_animations: BoolProperty(name="Include Animations", default=False)  # type: ignore

    def execute(self, context: Context):
        client_root = Path(bpy.path.abspath(self.client_folder))
        excel_root = Path(bpy.path.abspath(self.excel_data_folder))
        output_root = Path(bpy.path.abspath(self.output_folder))

        if self.mode == "ALL":
            jobs = []
            for mode in EXPORT_MODES:
                jobs.extend(build_jobs_for_mode(excel_root, mode))
        else:
            jobs = build_jobs_for_mode(excel_root, self.mode)

        if self.limit > 0:
            jobs = jobs[: self.limit]

        print(f"RF batch glB exporter: {len(jobs)} job(s), mode={self.mode}, dry_run={self.dry_run}")
        for job in jobs:
            print(f"  {job.name}: bone={job.bone}, meshes={len(job.meshes)}, animations={len(job.animations)}")

        if self.dry_run:
            self.report({"INFO"}, f"Dry run listed {len(jobs)} job(s) in the console.")
            return {"FINISHED"}

        exported = 0
        failed = 0
        output_root.mkdir(parents=True, exist_ok=True)
        log_path = output_root / f"rf_batch_glb_{self.mode.lower()}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        with tempfile.TemporaryDirectory(prefix="cbb_rf_batch_") as temp_dir:
            resolver = ClientAssetResolver(client_root, Path(temp_dir))
            with log_path.open("w", encoding="utf-8") as log:
                log.write("RF Batch glB Export Log\n")
                log.write(f"started: {datetime.now().isoformat(timespec='seconds')}\n")
                log.write(f"mode: {self.mode}\n")
                log.write(f"client_folder: {client_root}\n")
                log.write(f"excel_data_folder: {excel_root}\n")
                log.write(f"output_folder: {output_root}\n")
                log.write(f"include_animations: {self.include_animations}\n")
                log.write(f"jobs: {len(jobs)}\n\n")
                for index, job in enumerate(jobs, start=1):
                    log.write(f"[{index}/{len(jobs)}] {job.name}\n")
                    log.write(f"  kind: {job.kind}\n")
                    log.write(f"  requested_bone: {job.bone}\n")
                    log.write(f"  requested_meshes: {job.meshes}\n")
                    log.write(f"  requested_animations: {len(job.animations)}\n")
                    try:
                        ok, message = export_job(job, resolver, output_root / job.kind, self.include_animations, log)
                    except Exception as error:
                        ok = False
                        message = f"Exception: {error}"
                        log.write(traceback.format_exc())

                    if ok:
                        exported += 1
                        log.write(f"  result: OK {message}\n\n")
                    else:
                        failed += 1
                        line = f"FAILED {job.name}: {message}"
                        print(line)
                        log.write(f"  result: FAILED {message}\n\n")
                    log.flush()

                log.write(f"summary: exported={exported} failed={failed}\n")

        self.report({"INFO"}, f"Exported {exported} job(s), failed {failed}. Log: {log_path}")
        return {"FINISHED"}

    def invoke(self, context: Context, event: Event):
        return context.window_manager.invoke_props_dialog(self)


def menu_func_export(self, context):
    self.layout.operator(CBB_OT_BatchExportGLB.bl_idname, text="RF Batch glB Export")


def register():
    bpy.utils.register_class(CBB_OT_BatchExportGLB)
    bpy.types.TOPBAR_MT_file_export.append(menu_func_export)


def unregister():
    bpy.types.TOPBAR_MT_file_export.remove(menu_func_export)
    bpy.utils.unregister_class(CBB_OT_BatchExportGLB)

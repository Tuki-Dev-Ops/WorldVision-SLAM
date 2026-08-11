"""언리얼 없이 임포터를 **돌려 보려고** 두는 최소 스텁.

언리얼은 여기 설치되어 있지 않다. 그래도 임포터의 대부분은 엔진과 상관없는
일이다 - JSON 을 읽고, 좌표계를 바꾸고, 사각형을 만들고, 개수를 센다. 그
부분은 스텁으로 돌려서 확인할 수 있고, 실제로 거기서 버그가 난다.

확인하지 못하는 것은 에셋이 만들어지는지, 재질 그래프가 컴파일되는지,
화면에 어떻게 보이는지다. 그건 엔진이 있어야 한다.
"""

import math


class Vector:
    def __init__(self, x=0.0, y=0.0, z=0.0):
        self.x, self.y, self.z = float(x), float(y), float(z)

    def dot(self, o):
        return self.x * o.x + self.y * o.y + self.z * o.z

    def __repr__(self):
        return "V(%.2f, %.2f, %.2f)" % (self.x, self.y, self.z)


class Vector2D:
    def __init__(self, x=0.0, y=0.0):
        self.x, self.y = float(x), float(y)


class Rotator:
    def __init__(self, pitch=0.0, roll=0.0, yaw=0.0):
        self.pitch, self.roll, self.yaw = pitch, roll, yaw


class LinearColor:
    def __init__(self, r=0.0, g=0.0, b=0.0, a=1.0):
        self.r, self.g, self.b, self.a = r, g, b, a


class _Named:
    def __init__(self, name=""):
        self.name = name

    def set_editor_property(self, *_a, **_k):
        pass


class StaticMaterial:
    def __init__(self, material_interface=None):
        self.material_interface = material_interface


class Texture(_Named):
    address_x = None
    address_y = None


class TextureAddress:
    TA_CLAMP = "clamp"
    TA_WRAP = "wrap"


class MaterialProperty:
    MP_BASE_COLOR = "base"
    MP_ROUGHNESS = "rough"


class _Expr:
    texture = None
    constant = None
    r = 0.0


class Material(_Named):
    pass


class MaterialFactoryNew:
    pass


class StaticMeshFactoryNew:
    pass


class MaterialExpressionTextureSample:
    pass


class MaterialExpressionConstant3Vector:
    pass


class MaterialExpressionConstant:
    pass


class MaterialEditingLibrary:
    @staticmethod
    def create_material_expression(_m, _cls, _x, _y):
        return _Expr()

    @staticmethod
    def connect_material_property(*_a):
        pass

    @staticmethod
    def recompile_material(_m):
        pass


class StaticMeshDescription:
    """정점과 폴리곤을 세기만 한다. 그 수가 곧 만들어질 형상이다."""

    def __init__(self):
        self.verts = []
        self.instances = []
        self.polys = []
        self.uvs = {}

    def create_polygon_group(self):
        return 0

    def create_vertex(self):
        self.verts.append(None)
        return len(self.verts) - 1

    def set_vertex_position(self, vid, p):
        self.verts[vid] = p

    def create_vertex_instance(self, vid):
        self.instances.append(vid)
        return len(self.instances) - 1

    def set_vertex_instance_uv(self, vi, uv, _ch):
        self.uvs[vi] = uv

    def create_polygon(self, _pg, vis):
        self.polys.append(list(vis))


class StaticMesh(_Named):
    def __init__(self, name=""):
        _Named.__init__(self, name)
        self.descs = []

    def create_static_mesh_description(self):
        return StaticMeshDescription()

    def build_from_static_mesh_descriptions(self, mds):
        self.descs = list(mds)


class _Component:
    def __init__(self):
        self.mesh = None

    def set_static_mesh(self, m):
        self.mesh = m


class StaticMeshActor:
    pass


class _Actor:
    def __init__(self, loc, rot):
        self.location, self.rotation = loc, rot
        self.static_mesh_component = _Component()
        self.scale = Vector(1, 1, 1)
        self.label = ""
        self.folder = ""

    def set_actor_scale3d(self, s):
        self.scale = s

    def set_actor_label(self, s):
        self.label = s

    def set_folder_path(self, s):
        self.folder = s


# 검사기가 들여다보는 기록.
SPAWNED = []
MESHES = {}
LOG = []


class EditorLevelLibrary:
    @staticmethod
    def spawn_actor_from_class(_cls, loc, rot):
        a = _Actor(loc, rot)
        SPAWNED.append(a)
        return a


class EditorAssetLibrary:
    _assets = {}

    @staticmethod
    def load_asset(path):
        if path.startswith("/Engine/BasicShapes"):
            return StaticMesh(path)
        return EditorAssetLibrary._assets.get(path)

    @staticmethod
    def does_asset_exist(path):
        return path in EditorAssetLibrary._assets

    @staticmethod
    def delete_asset(path):
        EditorAssetLibrary._assets.pop(path, None)

    @staticmethod
    def save_asset(_path):
        pass

    @staticmethod
    def make_directory(_p):
        pass


class _AssetTools:
    def create_asset(self, name, path, cls, _factory):
        full = "%s/%s" % (path, name)
        obj = cls(full) if cls in (StaticMesh, Material) else cls()
        EditorAssetLibrary._assets[full] = obj
        if cls is StaticMesh:
            MESHES[full] = obj
        return obj

    def import_asset_tasks(self, tasks):
        for t in tasks:
            full = "%s/%s" % (t.destination_path, t.destination_name)
            EditorAssetLibrary._assets[full] = Texture(full)


class AssetToolsHelpers:
    @staticmethod
    def get_asset_tools():
        return _AssetTools()


class AssetImportTask:
    filename = ""
    destination_path = ""
    destination_name = ""
    automated = False
    replace_existing = False
    save = False


def log(msg):
    LOG.append(("log", msg))
    print(msg)


def log_warning(msg):
    LOG.append(("warn", msg))
    print("경고: %s" % msg)


def log_error(msg):
    LOG.append(("error", msg))
    print("오류: %s" % msg)

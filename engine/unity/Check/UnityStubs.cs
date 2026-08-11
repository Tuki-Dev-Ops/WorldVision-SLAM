// Unity 없이 임포터가 **컴파일되는지** 확인하려고 두는 최소 스텁.
//
// 에디터를 열었을 때 가장 흔히 깨지는 것은 컴파일 에러다. Unity 를 설치할
// 수 없어도 그건 여기서 없앨 수 있다 - 임포터가 실제로 쓰는 API 만 흉내
// 내면 되고, 시그니처가 어긋나면 컴파일러가 잡는다.
//
// 동작을 검증하지는 않는다. 이 스텁은 아무 것도 하지 않는다.
using System;

namespace UnityEngine
{
    public struct Color
    {
        public float r, g, b, a;
        public Color(float r, float g, float b) { this.r = r; this.g = g; this.b = b; this.a = 1f; }
        public Color(float r, float g, float b, float a) { this.r = r; this.g = g; this.b = b; this.a = a; }
    }

    public struct Vector3
    {
        public float x, y, z;
        public Vector3(float x, float y, float z) { this.x = x; this.y = y; this.z = z; }
        public float sqrMagnitude { get { return x * x + y * y + z * z; } }
        public Vector3 normalized { get { return this; } }
        public static Vector3 up { get { return new Vector3(0, 1, 0); } }
        public static Vector3 zero { get { return new Vector3(0, 0, 0); } }
        public static Vector3 operator +(Vector3 a, Vector3 b) { return new Vector3(a.x + b.x, a.y + b.y, a.z + b.z); }
        public static Vector3 operator *(Vector3 a, float s) { return new Vector3(a.x * s, a.y * s, a.z * s); }
    }

    public struct Quaternion
    {
        public static Quaternion identity { get { return default(Quaternion); } }
        public static Quaternion LookRotation(Vector3 f, Vector3 u) { return default(Quaternion); }
        public static Quaternion Euler(float x, float y, float z) { return default(Quaternion); }
        public static Quaternion operator *(Quaternion a, Quaternion b) { return default(Quaternion); }
    }

    public static class Mathf
    {
        public static float Clamp(float v, float lo, float hi) { return v < lo ? lo : (v > hi ? hi : v); }
        public static int Clamp(int v, int lo, int hi) { return v < lo ? lo : (v > hi ? hi : v); }
        public static float Clamp01(float v) { return Clamp(v, 0f, 1f); }
        public static float Max(float a, float b) { return a > b ? a : b; }
        public static float Min(float a, float b) { return a < b ? a : b; }
        public static float Sqrt(float v) { return (float)Math.Sqrt(v); }
    }

    public struct Bounds
    {
        public Bounds(Vector3 c, Vector3 s) { center = c; size = s; }
        public Vector3 center, size;
        public Vector3 extents { get { return size * 0.5f; } }
        public void Encapsulate(Vector3 p) { }
        public void Encapsulate(Bounds b) { }
    }

    public struct Rect
    {
        public Rect(float x, float y, float w, float h) { }
    }

    public enum LightType { Directional, Point, Spot }
    public enum CameraClearFlags { Skybox, SolidColor, Depth, Nothing }
    public enum FindObjectsSortMode { None, InstanceID }
    public enum TextureFormat { RGB24, RGBA32 }

    public class Light : Component
    {
        public LightType type;
        public float intensity, shadowStrength;
        public Color color;
        public LightShadows shadows;
    }
    public enum LightShadows { None, Hard, Soft }

    public class Camera : Component
    {
        public CameraClearFlags clearFlags;
        public Color backgroundColor;
        public float fieldOfView, nearClipPlane, farClipPlane, orthographicSize;
        public bool orthographic;
        public RenderTexture targetTexture;
        public void Render() { }
    }

    public class RenderTexture : Object
    {
        public RenderTexture(int w, int h, int depth) { }
        public int antiAliasing;
        public static RenderTexture active;
        public void Release() { }
    }

    public class Texture2D : Object
    {
        public Texture2D(int w, int h, TextureFormat f, bool mip) { }
        public void ReadPixels(Rect r, int x, int y) { }
        public void Apply() { }
        public byte[] EncodeToPNG() { return new byte[0]; }
    }

    public class Mesh : Object
    {
        public Rendering.IndexFormat indexFormat;
        public int subMeshCount;
        public void SetVertices(System.Collections.Generic.List<Vector3> v) { }
        public void SetColors(System.Collections.Generic.List<Color> c) { }
        public void SetTriangles(System.Collections.Generic.List<int> t, int sub) { }
        public void RecalculateNormals() { }
        public Bounds bounds;
    }

    public class MeshFilter : Component { public Mesh sharedMesh; }
    public class MeshRenderer : Renderer { }
    public class MeshCollider : Component { public Mesh sharedMesh; }
    public class BoxCollider : Component { }
    public class SphereCollider : Component { }
    public class CapsuleCollider : Component { }

    public class Object
    {
        public string name;
        public static T[] FindObjectsByType<T>(FindObjectsSortMode m) { return new T[0]; }
        public static void DestroyImmediate(Object o) { }
    }
    public class Shader : Object { public static Shader Find(string n) { return null; } }
    public class Material : Object { public Material(Shader s) { } public Color color; }
    public class Renderer : Component { public Material sharedMaterial; public Material[] sharedMaterials; public Bounds bounds; }
    public class Component : Object { public Transform transform; }

    public class Transform : Component
    {
        public Vector3 position, localScale;
        public Quaternion rotation;
        public void SetParent(Transform p, bool worldPositionStays) { }
    }

    public enum PrimitiveType { Cube, Sphere, Cylinder }

    public class GameObject : Object
    {
        public GameObject() { transform = new Transform(); }
        public GameObject(string n) { name = n; transform = new Transform(); }
        public Transform transform;
        public bool isStatic;
        public static GameObject CreatePrimitive(PrimitiveType t) { return new GameObject(); }
        public T GetComponent<T>() where T : Component, new() { return new T(); }
        public T AddComponent<T>() where T : Component, new() { return new T(); }
        public void SetActive(bool v) { }
    }

    public static class Debug
    {
        public static void Log(object o) { }
        public static void LogError(object o) { }
    }

    public static class Application { public static string dataPath { get { return "."; } } }

    public static class JsonUtility
    {
        public static T FromJson<T>(string s) { return default(T); }
    }
}

namespace UnityEngine.Rendering
{
    public enum IndexFormat { UInt16, UInt32 }
}

namespace UnityEditor
{
    using UnityEngine;
    public class MenuItem : Attribute { public MenuItem(string path) { } }
    public static class EditorUtility
    {
        public static string OpenFilePanel(string t, string d, string e) { return ""; }
    }
    public static class Undo
    {
        public static void RegisterCreatedObjectUndo(UnityEngine.Object o, string s) { }
    }
    public static class Selection { public static GameObject activeGameObject; }
    public static class EditorApplication { public static void Exit(int code) { } }
}

namespace UnityEditor.SceneManagement
{
    public struct Scene { }
    public static class EditorSceneManager
    {
        public static Scene GetActiveScene() { return default(Scene); }
        public static bool SaveScene(Scene s, string path) { return true; }
    }
}

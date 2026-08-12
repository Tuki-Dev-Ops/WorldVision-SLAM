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
        public static Color white { get { return new Color(1f, 1f, 1f); } }
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
        public static Vector3 right { get { return new Vector3(1, 0, 0); } }
        public static Vector3 forward { get { return new Vector3(0, 0, 1); } }
        public static Vector3 zero { get { return new Vector3(0, 0, 0); } }
        public static Vector3 operator -(Vector3 a, Vector3 b) { return new Vector3(a.x - b.x, a.y - b.y, a.z - b.z); }
        public static Vector3 operator -(Vector3 a) { return new Vector3(-a.x, -a.y, -a.z); }
        public static float Dot(Vector3 a, Vector3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
        public static float Distance(Vector3 a, Vector3 b) { return (a - b).magnitude; }
        public static Vector3 Lerp(Vector3 a, Vector3 b, float t) { return a; }
        public float magnitude { get { return (float)Math.Sqrt(sqrMagnitude); } }
        public static Vector3 operator +(Vector3 a, Vector3 b) { return new Vector3(a.x + b.x, a.y + b.y, a.z + b.z); }
        public static Vector3 operator *(Vector3 a, float s) { return new Vector3(a.x * s, a.y * s, a.z * s); }
    }

    public struct Quaternion
    {
        public static Quaternion identity { get { return default(Quaternion); } }
        public static Quaternion LookRotation(Vector3 f, Vector3 u) { return default(Quaternion); }
        public static Quaternion Euler(float x, float y, float z) { return default(Quaternion); }
        public Vector3 eulerAngles { get { return Vector3.zero; } }
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
        public static float Abs(float v) { return Math.Abs(v); }
        public static float Lerp(float a, float b, float t) { return a + (b - a) * t; }
        public static int Min(int a, int b) { return a < b ? a : b; }
        public static float Pow(float a, float b) { return (float)Math.Pow(a, b); }
        public static int RoundToInt(float v) { return (int)Math.Round(v); }
    }

    public struct Bounds
    {
        public Bounds(Vector3 c, Vector3 s) { center = c; size = s; }
        public Vector3 center, size;
        public Vector3 extents { get { return size * 0.5f; } }
        public Vector3 max { get { return center + extents; } }
        public Vector3 min { get { return center - extents; } }
        public void Encapsulate(Vector3 p) { }
        public void Encapsulate(Bounds b) { }
    }

    public struct Color32
    {
        public byte r, g, b, a;
        public Color32(byte r, byte g, byte b, byte a) { this.r = r; this.g = g; this.b = b; this.a = a; }
    }

    public struct Vector2
    {
        public float x, y;
        public Vector2(float x, float y) { this.x = x; this.y = y; }
    }

    public struct Rect
    {
        public Rect(float x, float y, float w, float h)
        { this.x = x; this.y = y; width = w; height = h; }
        public float x, y, width, height;
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
    public enum FullScreenMode { ExclusiveFullScreen, FullScreenWindow, MaximizedWindow, Windowed }
    public enum FogMode { Linear, Exponential, ExponentialSquared }
    public static class RenderSettings
    {
        public static Color ambientLight, fogColor;
        public static bool fog;
        public static FogMode fogMode;
        public static float fogStartDistance, fogEndDistance;
    }
    public class CharacterController : Component
    {
        public float height, radius, slopeLimit, stepOffset;
        public Vector3 center;
        public bool enabled, isGrounded;
        public void Move(Vector3 d) { }
    }
    public class MonoBehaviour : Component { public bool enabled; }
    public class RequireComponent : Attribute { public RequireComponent(Type t) { } }
    public static class Input
    {
        public static bool GetKeyDown(KeyCode k) { return false; }
        public static bool GetKey(KeyCode k) { return false; }
        public static float GetAxisRaw(string n) { return 0f; }
    }
    public enum KeyCode { Escape, C, F, H, L, Q, R, T, Space, LeftShift, RightShift, LeftControl,
                          Alpha1, Alpha2, Alpha3, Alpha4, LeftArrow, RightArrow,
                          LeftBracket, RightBracket }
    public enum CursorLockMode { None, Locked, Confined }
    public static class Cursor
    {
        public static CursorLockMode lockState;
        public static bool visible;
    }
    public static class Time { public static float deltaTime, unscaledDeltaTime; }
    public static class GUI
    {
        public static GUISkin skin;
        public static Color color;
        public static void Label(Rect r, string s, GUIStyle st) { }
        public static void DrawTexture(Rect r, Texture t) { }
    }
    public static class Screen { public static int width, height; }
    public class GUISkin { public GUIStyle label = new GUIStyle(); }
    public class GUIStyleState { public Color textColor; }
    public class GUIStyle
    {
        public GUIStyle() { }
        public GUIStyle(GUIStyle o) { }
        public int fontSize;
        public GUIStyleState normal = new GUIStyleState();
    }
    public class AudioListener : Component { }

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

    public class Texture2D : Texture
    {
        public Texture2D(int w, int h, TextureFormat f, bool mip) { }
        public Texture2D(int w, int h) { }
        public void SetPixel(int x, int y, Color c) { }
        public void ReadPixels(Rect r, int x, int y) { }
        public bool LoadImage(byte[] data) { return true; }
        public Color32[] GetPixels32() { return new Color32[0]; }
        public void SetPixels32(Color32[] c) { }
        public void Apply() { }
        public byte[] EncodeToPNG() { return new byte[0]; }
    }

    public class Mesh : Object
    {
        public Rendering.IndexFormat indexFormat;
        public int subMeshCount;
        public void SetVertices(System.Collections.Generic.List<Vector3> v) { }
        public void SetColors(System.Collections.Generic.List<Color> c) { }
        public void SetUVs(int ch, System.Collections.Generic.List<Vector2> uv) { }
        public Vector3[] vertices { get { return new Vector3[0]; } }
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
        public static T FindFirstObjectByType<T>() where T : Object { return null; }
        public static void DestroyImmediate(Object o) { }
    }
    public class Shader : Object { public static Shader Find(string n) { return null; } }
    public class Material : Object
    {
        public Material(Shader s) { }
        public Color color;
        public Texture mainTexture;
        public bool HasProperty(string n) { return true; }
        public bool enableInstancing;
        public void SetFloat(string n, float v) { }
    }
    public class Texture : Object { }
    public enum TextureWrapMode { Repeat, Clamp }
    public enum FilterMode { Point, Bilinear, Trilinear }
    public class Renderer : Component
    {
        public Material sharedMaterial;
        public Material[] sharedMaterials;
        public Bounds bounds;
        public bool enabled;
    }
    public class Component : Object
    {
        public Transform transform;
        public GameObject gameObject;
        public T GetComponent<T>() where T : Component, new() { return new T(); }
        public T GetComponentInChildren<T>() where T : Component, new() { return new T(); }
        public T[] GetComponentsInChildren<T>() where T : Component, new() { return new T[0]; }
    }

    public class Transform : Component
    {
        public Vector3 position, localScale;
        public Quaternion rotation;
        public Vector3 localPosition, eulerAngles, right, forward;
        public Quaternion localRotation;
        public void SetParent(Transform p, bool worldPositionStays, bool dummy) { }
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
        public string tag;
        public static GameObject Find(string n) { return null; }
        public void SetActive(bool v) { }
    }

    public static class Debug
    {
        public static void Log(object o) { }
        public static void LogError(object o) { }
        public static void LogWarning(object o) { }
    }

    public static class Application
    {
        public static string dataPath { get { return "."; } }
        public static void Quit() { }
    }

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
    public enum ImportAssetOptions { Default, ForceUpdate }
    public enum TextureImporterType { Default, NormalMap, Sprite }
    public class AssetImporter
    {
        public static AssetImporter GetAtPath(string p) { return null; }
        public void SaveAndReimport() { }
    }
    public enum TextureImporterNPOTScale { None, ToNearest, ToLarger, ToSmaller }
    public enum TextureImporterCompression { Uncompressed, Compressed, CompressedHQ }
    public class TextureImporter : AssetImporter
    {
        public TextureImporterNPOTScale npotScale;
        public TextureImporterCompression textureCompression;
        public TextureImporterType textureType;
        public TextureWrapMode wrapMode;
        public FilterMode filterMode;
        public bool mipmapEnabled;
        public int maxTextureSize;
    }
    public static class AssetDatabase
    {
        public static void ImportAsset(string p, ImportAssetOptions o) { }
        public static void CreateAsset(UnityEngine.Object o, string p) { }
        public static bool IsValidFolder(string p) { return true; }
        public static string CreateFolder(string parent, string name) { return ""; }
        public static bool DeleteAsset(string p) { return true; }
        public static void SaveAssets() { }
        public static T LoadAssetAtPath<T>(string p) where T : UnityEngine.Object { return null; }
    }
    public static class EditorApplication { public static void Exit(int code) { } }

    public enum BuildTarget { StandaloneWindows64 = 19 }
    [Flags] public enum BuildOptions { None = 0 }
    public struct BuildPlayerOptions
    {
        public string[] scenes;
        public string locationPathName;
        public BuildTarget target;
        public BuildOptions options;
    }
    public class EditorBuildSettingsScene
    {
        public EditorBuildSettingsScene(string path, bool enabled) { }
    }
    public static class EditorBuildSettings
    {
        public static EditorBuildSettingsScene[] scenes;
    }
    public static class BuildPipeline
    {
        public static UnityEditor.Build.Reporting.BuildReport BuildPlayer(BuildPlayerOptions o)
        {
            return new UnityEditor.Build.Reporting.BuildReport {
                summary = new UnityEditor.Build.Reporting.BuildSummary() };
        }
    }
    public static class PlayerSettings
    {
        public static string companyName, productName;
        public static int defaultScreenWidth, defaultScreenHeight;
        public static UnityEngine.FullScreenMode fullScreenMode;
        public static bool resizableWindow, runInBackground;
    }
}

namespace UnityEditor.Build.Reporting
{
    public enum BuildResult { Unknown, Succeeded, Failed, Cancelled }
    public class BuildSummary { public BuildResult result; public ulong totalSize; }
    public class BuildReport { public BuildSummary summary; }
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

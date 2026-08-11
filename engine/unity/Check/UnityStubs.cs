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
        public static float Max(float a, float b) { return a > b ? a : b; }
        public static float Min(float a, float b) { return a < b ? a : b; }
    }

    public class Object { public string name; }
    public class Shader : Object { public static Shader Find(string n) { return null; } }
    public class Material : Object { public Material(Shader s) { } public Color color; }
    public class Renderer : Component { public Material sharedMaterial; }
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
}

// 세운 장면 안을 직접 걸어 다니게 한다.
//
// 임포트한 씬은 그때까지 **보기만 하는 것** 이었다. 위에서 한 장 찍어 배치가
// 맞는지는 확인했지만, 그것은 그림이지 돌아다닐 수 있는 세계가 아니다. 노면에
// MeshCollider 를 붙이고 건물에 BoxCollider 를 붙인 이유가 여기서 쓰인다.
//
// 걷기가 기본이다. 날기는 F 로 켠다 - 지붕 위나 골목 안쪽을 보려면 필요한데,
// 처음부터 날고 있으면 바닥이 있는지 없는지가 확인되지 않는다.
using UnityEngine;

namespace WorldVision
{
    [RequireComponent(typeof(CharacterController))]
    public class Player : MonoBehaviour
    {
        public float walk = 6f;          // m/s. 사람보다 조금 빠르다
        public float sprint = 22f;       // 250 m 짜리 길을 건너려면 필요하다
        public float fly = 18f;
        public float look = 2.2f;
        public float gravity = 20f;
        public float jump = 6f;

        CharacterController cc;
        Camera cam;
        float pitch, yaw;
        float vy;
        bool flying;
        bool help = true;

        void Start()
        {
            cc = GetComponent<CharacterController>();
            cam = GetComponentInChildren<Camera>();
            yaw = transform.eulerAngles.y;
            Cursor.lockState = CursorLockMode.Locked;
            Cursor.visible = false;
        }

        void Update()
        {
            // 마우스를 놓아 주는 길이 없으면 창을 벗어날 수가 없다.
            if (Input.GetKeyDown(KeyCode.Escape))
            {
                Cursor.lockState = Cursor.lockState == CursorLockMode.Locked
                                 ? CursorLockMode.None : CursorLockMode.Locked;
                Cursor.visible = Cursor.lockState != CursorLockMode.Locked;
            }
            if (Input.GetKeyDown(KeyCode.F)) flying = !flying;
            if (Input.GetKeyDown(KeyCode.H)) help = !help;
            if (Input.GetKeyDown(KeyCode.Q)) Application.Quit();

            if (Cursor.lockState == CursorLockMode.Locked)
            {
                yaw += Input.GetAxisRaw("Mouse X") * look;
                pitch = Mathf.Clamp(pitch - Input.GetAxisRaw("Mouse Y") * look,
                                    -85f, 85f);
            }
            transform.rotation = Quaternion.Euler(0f, yaw, 0f);
            if (cam != null) cam.transform.localRotation = Quaternion.Euler(pitch, 0f, 0f);

            float f = Input.GetAxisRaw("Vertical");
            float r = Input.GetAxisRaw("Horizontal");
            bool fast = Input.GetKey(KeyCode.LeftShift) || Input.GetKey(KeyCode.RightShift);

            if (flying)
            {
                // 날 때는 보는 쪽으로 간다. 수평으로만 가면 지붕을 못 넘는다.
                Vector3 dir = (cam != null ? cam.transform.forward : transform.forward) * f
                            + transform.right * r;
                if (Input.GetKey(KeyCode.Space)) dir += Vector3.up;
                if (Input.GetKey(KeyCode.LeftControl)) dir -= Vector3.up;
                cc.Move(dir.normalized * (fly * (fast ? 3f : 1f)) * Time.deltaTime);
                vy = 0f;
                return;
            }

            Vector3 move = (transform.forward * f + transform.right * r).normalized
                         * (fast ? sprint : walk);
            if (cc.isGrounded)
            {
                // 살짝 눌러 둔다. 0 이면 경사에서 매 프레임 공중에 뜬다.
                vy = Input.GetKeyDown(KeyCode.Space) ? jump : -2f;
            }
            else
            {
                vy -= gravity * Time.deltaTime;
            }
            move.y = vy;
            cc.Move(move * Time.deltaTime);

            // **바닥을 잃으면 되돌린다.** 노면은 관측된 곳에만 있으므로 가장자리
            // 밖으로 나가면 끝없이 떨어진다. 떨어지는 것은 버그처럼 보이지만
            // 실은 지도의 경계다 - 그 자리에서 멈추고 돌려보내는 편이 낫다.
            if (transform.position.y < -60f)
            {
                cc.enabled = false;
                transform.position = new Vector3(transform.position.x, 20f,
                                                 transform.position.z);
                cc.enabled = true;
                vy = 0f;
            }
        }

        void OnGUI()
        {
            if (!help) return;
            var st = new GUIStyle(GUI.skin.label);
            st.fontSize = 15;
            st.normal.textColor = new Color(0.85f, 0.95f, 1f);
            GUI.Label(new Rect(14, 10, 900, 130),
                "WorldVision-SLAM  —  인식한 장면을 그대로 세운 것\n" +
                "WASD 이동   Shift 질주   Space 점프   마우스 시점\n" +
                "F 날기/걷기   Esc 마우스 풀기   H 이 안내 끄기   Q 종료\n" +
                (flying ? "지금: 날기" : "지금: 걷기 (노면 충돌체 위)"), st);
        }
    }
}

using System.Collections.Generic;
using UnityEngine;
using UnityEngine.XR;


public class HeartVRController : MonoBehaviour
{
    [Header("播放器（留空自動找）")]
    public Gaussian4DPlayer player;

    [Header("旋轉/平移目標（留空自動用 player 的 Transform）")]
    public Transform heartTransform;

    [Header("自旋速度（度/秒，板機推滿時）")]
    public float spinSpeed = 90f;

    [Header("平移速度（單位/秒）")]
    public float moveSpeed = 0.5f;

    [Header("自旋方向反向")]
    public bool invertLeftSpin = false;   
    public bool invertRightSpin = false;  

    private Gaussian4DPlayer _player;
    private InputDevice _left, _right;
    private Vector3 _localPivot;
    private bool _pivotReady;
    private bool _prevMenu;

    void Awake()
    {
        _player = player;
        if (_player == null) _player = GetComponent<Gaussian4DPlayer>();
        if (_player == null) _player = GetComponentInChildren<Gaussian4DPlayer>();
        if (_player == null) _player = FindObjectOfType<Gaussian4DPlayer>();
        if (_player == null)
            Debug.LogWarning("[HeartVRController] can't find Gaussian4DPlayer");
        if (heartTransform == null && _player != null)
            heartTransform = _player.transform;
    }

    void Update()
    {
        TryAcquireControllers();
        EnsurePivot();
        HandleVRSpin();
        HandleVRPlayPause();
        HandleKeyboard();
    }

    void TryAcquireControllers()
    {
        if (!_left.isValid)  _left  = GetDevice(InputDeviceCharacteristics.Left);
        if (!_right.isValid) _right = GetDevice(InputDeviceCharacteristics.Right);
    }

    InputDevice GetDevice(InputDeviceCharacteristics hand)
    {
        var list = new List<InputDevice>();
        InputDevices.GetDevicesWithCharacteristics(
            hand | InputDeviceCharacteristics.Controller, list);
        return list.Count > 0 ? list[0] : default;
    }

    void EnsurePivot()
    {
        if (_pivotReady || _player == null || heartTransform == null) return;
        if (_player.TryGetHeartWorldCenter(out Vector3 c))
        {
            _localPivot = heartTransform.InverseTransformPoint(c);
            _pivotReady = true;
        }
    }

    Vector3 PivotWorld => _pivotReady
        ? heartTransform.TransformPoint(_localPivot)
        : (heartTransform ? heartTransform.position : Vector3.zero);

    void SpinY(float deg) { if (heartTransform) heartTransform.RotateAround(PivotWorld, Vector3.up, deg); }
    void SpinX(float deg) { if (heartTransform) heartTransform.RotateAround(PivotWorld, Vector3.right, deg); }
    void Move(Vector3 dir) { if (heartTransform) heartTransform.position += dir * moveSpeed * Time.deltaTime; }

    void HandleVRSpin()
    {
        float tl = 0f, tr = 0f;
        if (_left.isValid)  _left.TryGetFeatureValue(CommonUsages.trigger, out tl);
        if (_right.isValid) _right.TryGetFeatureValue(CommonUsages.trigger, out tr);

        if (tl > 0.05f) SpinY((invertLeftSpin  ? -1f : 1f) * tl * spinSpeed * Time.deltaTime);
        if (tr > 0.05f) SpinX((invertRightSpin ? -1f : 1f) * tr * spinSpeed * Time.deltaTime);
    }

    void HandleVRPlayPause()
    {
        bool menu = false;
        if (_right.isValid) _right.TryGetFeatureValue(CommonUsages.menuButton, out menu);
        if (!menu && _left.isValid) _left.TryGetFeatureValue(CommonUsages.menuButton, out menu);

        if (menu && !_prevMenu && _player != null)
            _player.SetPlaying(!_player.IsPlaying);
        _prevMenu = menu;
    }

    void HandleKeyboard()
    {
        Vector3 d = Vector3.zero;
        if (Input.GetKey(KeyCode.UpArrow)    || Input.GetKey(KeyCode.W)) d += Vector3.forward;
        if (Input.GetKey(KeyCode.DownArrow)  || Input.GetKey(KeyCode.S)) d += Vector3.back;
        if (Input.GetKey(KeyCode.LeftArrow)  || Input.GetKey(KeyCode.A)) d += Vector3.left;
        if (Input.GetKey(KeyCode.RightArrow) || Input.GetKey(KeyCode.D)) d += Vector3.right;
        if (d != Vector3.zero) Move(d.normalized);

        if (Input.GetKey(KeyCode.Q)) SpinY( spinSpeed * Time.deltaTime);
        if (Input.GetKey(KeyCode.E)) SpinY(-spinSpeed * Time.deltaTime);
        if (Input.GetKey(KeyCode.R)) SpinX( spinSpeed * Time.deltaTime);
        if (Input.GetKey(KeyCode.F)) SpinX(-spinSpeed * Time.deltaTime);

        if (_player != null && Input.GetKeyDown(KeyCode.Space))
            _player.SetPlaying(!_player.IsPlaying);
    }
}

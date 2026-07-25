using System.Collections.Generic;
using UnityEngine;
using UnityEngine.XR;

public class GaussianDiag : MonoBehaviour
{
    public Gaussian4DPlayer player;
    private int _updateCount;
    private float _logTimer;

    void Awake()
    {
        if (player == null) player = GetComponent<Gaussian4DPlayer>();
        if (player == null) player = FindObjectOfType<Gaussian4DPlayer>();
    }

    void Update()
    {
        _updateCount++;
        _logTimer += Time.unscaledDeltaTime;
        if (_logTimer >= 1f)
        {
            _logTimer = 0f;
            Debug.Log(Line());
        }
    }

    string Trig()
    {
        float tl = -1f, tr = -1f;
        var l = new List<InputDevice>(); var r = new List<InputDevice>();
        InputDevices.GetDevicesWithCharacteristics(InputDeviceCharacteristics.Left | InputDeviceCharacteristics.Controller, l);
        InputDevices.GetDevicesWithCharacteristics(InputDeviceCharacteristics.Right | InputDeviceCharacteristics.Controller, r);
        if (l.Count > 0) l[0].TryGetFeatureValue(CommonUsages.trigger, out tl);
        if (r.Count > 0) r[0].TryGetFeatureValue(CommonUsages.trigger, out tr);
        return $"dev L{l.Count} R{r.Count}  trig {tl:F2}/{tr:F2}";
    }

    string Line()
    {
        string p = player == null ? "player=NULL"
            : $"loaded={player.IsLoaded} play={player.IsPlaying} frame={player.CurrentFrame}/{player.TotalFrames} layers={player.LayerCount}";
        return $"[Diag] updates={_updateCount} dt={Time.deltaTime * 1000f:F1}ms scale={Time.timeScale} | {p} | {Trig()}";
    }

    void OnGUI()
    {
        var style = new GUIStyle(GUI.skin.box)
        { alignment = TextAnchor.UpperLeft, fontSize = 16, normal = { textColor = Color.green } };
        GUI.Box(new Rect(10, 150, 560, 90), Line(), style);
    }
}

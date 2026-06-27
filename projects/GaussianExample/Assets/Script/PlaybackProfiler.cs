using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Text;
using UnityEngine;
using UnityEngine.Profiling;

public class PlaybackProfiler : MonoBehaviour
{
    [Header("標籤（寫進 Console / CSV，方便辨識，如 Before / After）")]
    public string runLabel = "Run";

    [Header("關聯播放器（留空自動找）")]
    public Gaussian4DPlayer player;

    [Header("量測：載入完成後延遲幾秒才開始（讓 GPU 暖機）")]
    public float startDelay = 2f;
    [Header("量測：持續幾秒")]
    public float measureDuration = 30f;
    [Header("量測期間 Console 每幾秒印一次即時值")]
    public float logInterval = 5f;

    [Header("左上角疊圖（Game view / 桌面鏡像可見）")]
    public bool showOverlay = true;

    private readonly Queue<float> _avgWin = new Queue<float>();
    private float _avgSum, _liveAvg, _liveLow;

    private enum State { WaitingLoad, Delay, Measuring, Done }
    private State _state = State.WaitingLoad;
    private float _delayTimer, _measureTimer, _logTimer;
    private readonly List<float> _samples = new List<float>();

    private long _gfxStart = -1, _gfxLoaded = -1;
    private bool _loadCaptured;

    void Awake()
    {
        if (player == null) player = FindObjectOfType<Gaussian4DPlayer>();
    }

    void Start()
    {
        _gfxStart = Profiler.GetAllocatedMemoryForGraphicsDriver();
        Debug.Log($"[Profiler:{runLabel}] GPU={SystemInfo.graphicsDeviceName} | " +
                  $"VRAM總量={SystemInfo.graphicsMemorySize}MB | 起始顯存={MB(_gfxStart):F1}MB");
    }

    void Update()
    {
        float dt = Time.unscaledDeltaTime;
        if (dt <= 0f) return;
        float fps = 1f / dt;

        _avgWin.Enqueue(dt); _avgSum += dt;
        while (_avgSum > 1f && _avgWin.Count > 1) _avgSum -= _avgWin.Dequeue();
        _liveAvg = _avgWin.Count / _avgSum;

        if (!_loadCaptured && player != null && player.IsLoaded)
        {
            _gfxLoaded = Profiler.GetAllocatedMemoryForGraphicsDriver();
            _loadCaptured = true;
            Debug.Log($"[Profiler:{runLabel}] 載入後顯存={MB(_gfxLoaded):F1}MB | " +
                      $"序列增量={MB(_gfxLoaded - _gfxStart):F1}MB | 幀數={(player ? player.TotalFrames : 0)}");
        }

        if (Input.GetKeyDown(KeyCode.B)) StartMeasure(); 

        switch (_state)
        {
            case State.WaitingLoad:
                if (player == null || player.IsLoaded) { _state = State.Delay; _delayTimer = 0f; }
                break;

            case State.Delay:
                _delayTimer += dt;
                if (_delayTimer >= startDelay) StartMeasure();
                break;

            case State.Measuring:
                _samples.Add(fps);
                _measureTimer += dt;
                _logTimer += dt;
                if (_logTimer >= logInterval)
                {
                    _logTimer = 0f;
                    _liveLow = Percentile1(_samples);
                    Debug.Log($"[Profiler:{runLabel}] 量測中 {_measureTimer:F0}/{measureDuration:F0}s | " +
                              $"即時avg={_liveAvg:F1} | 1%Low(至今)={_liveLow:F1} | " +
                              $"VRAM={MB(Profiler.GetAllocatedMemoryForGraphicsDriver()):F0}MB");
                }
                if (_measureTimer >= measureDuration) FinishMeasure();
                break;
        }
    }

    private void StartMeasure()
    {
        _samples.Clear(); _measureTimer = 0f; _logTimer = 0f;
        _state = State.Measuring;
        Debug.Log($"[Profiler:{runLabel}] === 開始量測 {measureDuration:F0} 秒 ===");
    }

    private void FinishMeasure()
    {
        _state = State.Done;
        if (_samples.Count == 0) return;

        var sorted = new List<float>(_samples); sorted.Sort();
        int n = sorted.Count;
        double sum = 0; foreach (var f in sorted) sum += f;
        float avg = (float)(sum / n);
        float low1 = sorted[Mathf.Clamp(Mathf.FloorToInt(n * 0.01f), 0, n - 1)];
        float min = sorted[0], max = sorted[n - 1];
        float vramTotal = MB(_gfxLoaded > 0 ? _gfxLoaded : Profiler.GetAllocatedMemoryForGraphicsDriver());
        float vramSeq = MB((_gfxLoaded > 0 ? _gfxLoaded : _gfxStart) - _gfxStart);

        var sb = new StringBuilder();
        sb.AppendLine($"================ VR 效能統計 [{runLabel}] ================");
        sb.AppendLine($"GPU             : {SystemInfo.graphicsDeviceName}");
        sb.AppendLine($"序列幀數         : {(player ? player.TotalFrames : 0)}");
        sb.AppendLine($"量測時長 / 取樣   : {measureDuration:F0}s / {n} 幀");
        sb.AppendLine($"FPS 平均         : {avg:F1}");
        sb.AppendLine($"FPS 1% Low       : {low1:F1}");
        sb.AppendLine($"FPS 最低 / 最高   : {min:F1} / {max:F1}");
        sb.AppendLine($"VRAM 載入後總量   : {vramTotal:F0} MB");
        sb.AppendLine($"VRAM 序列增量     : {vramSeq:F0} MB");
        sb.AppendLine($"========================================================");
        Debug.Log(sb.ToString());

        WriteCsv(n, avg, low1, min, max, vramTotal, vramSeq);
    }

    private void WriteCsv(int samples, float avg, float low1, float min, float max, float vramTotal, float vramSeq)
    {
        try
        {
            string path = Path.Combine(Application.persistentDataPath, "vr_perf.csv");
            bool header = !File.Exists(path);
            var c = CultureInfo.InvariantCulture;
            using (var w = new StreamWriter(path, true, Encoding.UTF8))
            {
                if (header)
                    w.WriteLine("label,gpu,frames,samples,duration_s,avg_fps,low1pct_fps,min_fps,max_fps,vram_total_mb,vram_seq_mb");
                w.WriteLine(string.Join(",",
                    runLabel, "\"" + SystemInfo.graphicsDeviceName + "\"",
                    (player ? player.TotalFrames : 0).ToString(c), samples.ToString(c),
                    measureDuration.ToString("F0", c), avg.ToString("F2", c), low1.ToString("F2", c),
                    min.ToString("F2", c), max.ToString("F2", c),
                    vramTotal.ToString("F0", c), vramSeq.ToString("F0", c)));
            }
            Debug.Log($"[Profiler:{runLabel}] 統計已寫入：{path}");
        }
        catch (System.Exception e)
        {
            Debug.LogWarning($"[Profiler:{runLabel}] 寫檔失敗：{e.Message}");
        }
    }

    private static float Percentile1(List<float> data)
    {
        if (data.Count < 10) return 0f;
        var s = new List<float>(data); s.Sort();
        return s[Mathf.Clamp(Mathf.FloorToInt(s.Count * 0.01f), 0, s.Count - 1)];
    }

    void OnGUI()
    {
        if (!showOverlay) return;
        var style = new GUIStyle(GUI.skin.box)
        {
            alignment = TextAnchor.UpperLeft, fontSize = 20, normal = { textColor = Color.white }
        };
        long gfxNow = Profiler.GetAllocatedMemoryForGraphicsDriver();
        string seq = _loadCaptured ? $"  (序列 +{MB(_gfxLoaded - _gfxStart):F0}MB)" : "";
        string st = _state switch
        {
            State.WaitingLoad => "等待載入",
            State.Delay       => "準備量測…",
            State.Measuring   => $"量測中 {_measureTimer:F0}/{measureDuration:F0}s",
            State.Done        => "量測完成（按 B 重測）",
            _ => ""
        };
        string txt =
            $"[{runLabel}] {st}\n" +
            $"FPS 平均 : {_liveAvg:F1}\n" +
            $"1% Low  : {_liveLow:F1}\n" +
            $"VRAM    : {MB(gfxNow):F0} MB{seq}";
        GUI.Box(new Rect(12, 12, 380, 116), txt, style);
    }

    private static float MB(long bytes) => bytes / (1024f * 1024f);
}

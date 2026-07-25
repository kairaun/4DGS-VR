using GaussianSplatting.Runtime;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Gaussian4DPlayer : MonoBehaviour
{
    [System.Serializable]
    public class Layer
    {
        public string name = "Layer";
        public GaussianSplatAsset[] frames;
        public bool visibleOnStart = true;
    }

    [Header("4DGS renderer template (GaussianSplatRenderer in scene)")]
    public GaussianSplatRenderer templateRenderer;

    [Header("Anatomical layers, each an ordered .asset sequence")]
    public Layer[] layers;

    [Header("Playback rate")]
    public float targetFPS = 30f;

    [Header("Instantiations per yield during preload")]
    public int loadBatchPerFrame = 4;

    [Header("Auto play once loaded")]
    public bool autoPlayWhenLoaded = true;

    private readonly List<GaussianSplatRenderer[]> pool = new List<GaussianSplatRenderer[]>();
    private readonly List<bool> layerVisible = new List<bool>();
    private float timer = 0f;
    private int currentFrame = 0;
    private int frameCount = 0;
    private bool autoPlay = false;
    private bool isLoaded = false;

    public int CurrentFrame => currentFrame;
    public int TotalFrames => frameCount;
    public int LayerCount => pool.Count;
    public int TotalRenderers
    {
        get { int s = 0; foreach (var a in pool) s += a.Length; return s; }
    }
    public bool IsPlaying => autoPlay;
    public bool IsLoaded => isLoaded;
    public float LoadProgress
    {
        get
        {
            int total = 0; foreach (var l in layers) total += l.frames != null ? l.frames.Length : 0;
            if (total == 0) return 1f;
            int done = 0; foreach (var a in pool) done += a.Length;
            return (float)done / total;
        }
    }

    public bool IsLayerVisible(int i) => i >= 0 && i < layerVisible.Count && layerVisible[i];
    public string LayerName(int i) => i >= 0 && i < layers.Length ? layers[i].name : "";

    void Start()
    {
        if (templateRenderer == null || layers == null || layers.Length == 0)
        {
            Debug.LogError("[Gaussian4DPlayer] templateRenderer or layers not set.");
            return;
        }
        StartCoroutine(LoadRoutine());
    }

    private IEnumerator LoadRoutine()
    {
        templateRenderer.gameObject.SetActive(false);

        foreach (var layer in layers)
        {
            var arr = new GaussianSplatRenderer[layer.frames != null ? layer.frames.Length : 0];
            frameCount = Mathf.Max(frameCount, arr.Length);
            for (int i = 0; i < arr.Length; i++)
            {
                GameObject clone = Instantiate(templateRenderer.gameObject, this.transform);
                clone.name = $"{layer.name}_Frame_{i}";
                GaussianSplatRenderer gs = clone.GetComponent<GaussianSplatRenderer>();
                gs.m_Asset = layer.frames[i];
                gs.m_IsActiveFrame = (i == 0) && layer.visibleOnStart;
                clone.SetActive(true);
                arr[i] = gs;
                if ((i + 1) % Mathf.Max(1, loadBatchPerFrame) == 0)
                    yield return null;
            }
            pool.Add(arr);
            layerVisible.Add(layer.visibleOnStart);
        }

        isLoaded = true;
        autoPlay = autoPlayWhenLoaded;
        Debug.Log($"[Gaussian4DPlayer] loaded {pool.Count} layers x {frameCount} frames = {TotalRenderers} renderers.");
    }

    void Update()
    {
        if (!isLoaded || !autoPlay || frameCount == 0) return;
        timer += Time.deltaTime;
        if (timer >= 1f / targetFPS)
        {
            timer -= 1f / targetFPS;
            AdvanceFrame(1);
        }
    }

    public void SetPlaying(bool playing)
    {
        if (!isLoaded) return;
        autoPlay = playing;
        if (playing) timer = 0f;
    }

    public void StepFrames(int delta)
    {
        if (!isLoaded || frameCount == 0) return;
        AdvanceFrame(delta);
    }

    private void AdvanceFrame(int delta)
    {
        for (int L = 0; L < pool.Count; L++)
        {
            if (!layerVisible[L]) continue;
            var arr = pool[L];
            if (arr.Length > 0) arr[currentFrame % arr.Length].m_IsActiveFrame = false;
        }
        currentFrame = ((currentFrame + delta) % frameCount + frameCount) % frameCount;
        for (int L = 0; L < pool.Count; L++)
        {
            if (!layerVisible[L]) continue;
            var arr = pool[L];
            if (arr.Length > 0) arr[currentFrame % arr.Length].m_IsActiveFrame = true;
        }
    }

    public void SetLayerVisible(int i, bool visible)
    {
        if (!isLoaded || i < 0 || i >= pool.Count) return;
        if (layerVisible[i] == visible) return;
        layerVisible[i] = visible;
        var arr = pool[i];
        if (arr.Length > 0) arr[currentFrame % arr.Length].m_IsActiveFrame = visible;
    }

    public void ToggleLayer(int i)
    {
        if (i >= 0 && i < layerVisible.Count) SetLayerVisible(i, !layerVisible[i]);
    }

    public bool TryGetHeartWorldCenter(out Vector3 center)
    {
        center = transform.position;
        for (int L = 0; L < pool.Count; L++)
        {
            if (!layerVisible[L]) continue;
            var arr = pool[L];
            if (arr.Length == 0) continue;
            var r = arr[currentFrame % arr.Length];
            if (r == null || r.m_Asset == null) continue;
            Vector3 localC = (r.m_Asset.boundsMin + r.m_Asset.boundsMax) * 0.5f;
            center = r.transform.TransformPoint(localC);
            return true;
        }
        return false;
    }
}

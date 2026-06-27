using GaussianSplatting.Runtime;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;


public class Gaussian4DPlayer : MonoBehaviour
{
    [Header("4DGS 渲染器模板（場景中的 GaussianSplatRenderer）")]
    public GaussianSplatRenderer templateRenderer;

    [Header("所有幀的 4DGS .asset")]
    public GaussianSplatAsset[] sequenceFrames;

    [Header("播放速率")]
    public float targetFPS = 30f;

    [Header("載入：每幀實例化幾個（越小越不卡，但載入越久）")]
    public int loadBatchPerFrame = 4;

    [Header("載入完成前是否自動開始播放")]
    public bool autoPlayWhenLoaded = true;

    private readonly List<GaussianSplatRenderer> renderersPool = new List<GaussianSplatRenderer>();
    private float timer = 0f;
    private int currentFrame = 0;
    private bool autoPlay = false;
    private bool isLoaded = false;

    public int CurrentFrame => currentFrame;
    public int TotalFrames  => renderersPool.Count;
    public bool IsPlaying   => autoPlay;
    public bool IsLoaded    => isLoaded;
    public float LoadProgress => sequenceFrames == null || sequenceFrames.Length == 0
        ? 1f : (float)renderersPool.Count / sequenceFrames.Length;

    void Start()
    {
        if (templateRenderer == null || sequenceFrames == null || sequenceFrames.Length == 0)
        {
            Debug.LogError("[Gaussian4DPlayer] templateRenderer 或 sequenceFrames 未設定。");
            return;
        }
        StartCoroutine(LoadRoutine());
    }

    private IEnumerator LoadRoutine()
    {
        templateRenderer.gameObject.SetActive(false); 
        Debug.Log($"[Gaussian4DPlayer] 開始分批預載 {sequenceFrames.Length} 幀");

        for (int i = 0; i < sequenceFrames.Length; i++)
        {
            GameObject cloneObj = Instantiate(templateRenderer.gameObject, this.transform);
            cloneObj.name = $"Heart_Frame_{i}";

            GaussianSplatRenderer gs = cloneObj.GetComponent<GaussianSplatRenderer>();
            gs.m_Asset = sequenceFrames[i];
            gs.m_IsActiveFrame = (i == 0);  

            cloneObj.SetActive(true);        
            renderersPool.Add(gs);

            if ((i + 1) % Mathf.Max(1, loadBatchPerFrame) == 0)
                yield return null;
        }

        isLoaded = true;
        autoPlay = autoPlayWhenLoaded;
        Debug.Log($"[Gaussian4DPlayer] 預載完成，共 {renderersPool.Count} 幀。");
    }

    void Update()
    {
        if (!isLoaded || !autoPlay || renderersPool.Count == 0) return;

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
        if (!isLoaded || renderersPool.Count == 0) return;
        AdvanceFrame(delta);
    }

    private void AdvanceFrame(int delta)
    {
        renderersPool[currentFrame].m_IsActiveFrame = false;
        int n = renderersPool.Count;
        currentFrame = ((currentFrame + delta) % n + n) % n;
        renderersPool[currentFrame].m_IsActiveFrame = true;
    }

    public bool TryGetHeartWorldCenter(out Vector3 center)
    {
        center = transform.position;
        if (renderersPool.Count == 0) return false;
        var r = renderersPool[currentFrame];
        if (r == null || r.m_Asset == null) return false;
        Vector3 localC = (r.m_Asset.boundsMin + r.m_Asset.boundsMax) * 0.5f;
        center = r.transform.TransformPoint(localC);
        return true;
    }
}

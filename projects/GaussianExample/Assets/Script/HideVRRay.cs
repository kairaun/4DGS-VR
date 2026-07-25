using UnityEngine;

public class HideVRRay : MonoBehaviour
{
    [Header("Also disable XR ray interactors if present")]
    public bool disableInteractors = true;

    void Start() { Hide(); }
    void OnEnable() { Hide(); }

    void Hide()
    {
        foreach (var lr in FindObjectsOfType<LineRenderer>(true))
            lr.enabled = false;

        if (!disableInteractors) return;
        foreach (var mb in FindObjectsOfType<MonoBehaviour>(true))
        {
            var t = mb.GetType().Name;
            if (t.Contains("RayInteractor") || t.Contains("InteractorLineVisual"))
                mb.enabled = false;
        }
    }
}

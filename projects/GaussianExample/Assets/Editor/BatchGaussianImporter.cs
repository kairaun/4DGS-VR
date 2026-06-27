using GaussianSplatting.Editor; 
using System.IO;
using System.Reflection;
using UnityEditor;
using UnityEngine;


public class BatchGaussianImporter : EditorWindow
{
    private string plyDirectory = "Assets/PLY/gaussian_pertimestamp";
    private string outputDirectory = "Assets/PLY/BatchOutput";

    [MenuItem("Tools/Import 4DGS PLY sequence")]
    public static void ShowWindow() => GetWindow<BatchGaussianImporter>("Import PLY");

    void OnGUI()
    {
        GUILayout.Label("PLY → GaussianSplatAsset", EditorStyles.boldLabel);
        plyDirectory = EditorGUILayout.TextField("Import folder (PLY)", plyDirectory);
        outputDirectory = EditorGUILayout.TextField("Output folder (Asset)", outputDirectory);

        if (GUILayout.Button("Start", GUILayout.Height(40)))
            BatchProcess(plyDirectory, outputDirectory);
    }

    private void BatchProcess(string inputFolder, string outputFolder)
    {
        if (!Directory.Exists(inputFolder))
        {
            Debug.LogError($"Cant't find Folder: {inputFolder}");
            return;
        }

        string[] plyFiles = Directory.GetFiles(inputFolder, "*.ply");
        if (plyFiles.Length == 0)
        {
            Debug.LogWarning("No .ply！");
            return;
        }

        var creator = ScriptableObject.CreateInstance<GaussianSplatAssetCreator>();
        var type = typeof(GaussianSplatAssetCreator);
        var inputFileField    = type.GetField("m_InputFile",    BindingFlags.NonPublic | BindingFlags.Instance);
        var outputFolderField = type.GetField("m_OutputFolder", BindingFlags.NonPublic | BindingFlags.Instance);
        var createAssetMethod = type.GetMethod("CreateAsset",   BindingFlags.NonPublic | BindingFlags.Instance);

        if (inputFileField == null || outputFolderField == null || createAssetMethod == null)
        {
            Debug.LogError("Can't find aras-p");
            return;
        }

        int count = 0;
        foreach (string file in plyFiles)
        {
            count++;
            string safe = file.Replace("\\", "/");
            inputFileField.SetValue(creator, safe);
            outputFolderField.SetValue(creator, outputFolder);

            Debug.Log($"[{count}/{plyFiles.Length}] Switch: {Path.GetFileName(safe)}");
            createAssetMethod.Invoke(creator, null);
        }

        AssetDatabase.Refresh();
        Debug.Log($"Switch complete：{plyFiles.Length} file → {outputFolder}</color>");
    }
}

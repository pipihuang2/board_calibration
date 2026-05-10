# circle_board

这是一个基于 `PyQt6 + OpenCV + NumPy` 的椭圆长短轴比分析工具。

## 本地运行

在项目根目录执行：

```powershell
python -m pip install -r requirements.txt
python main.py
```

## GitHub 自动打包 EXE

项目已经配置了 GitHub Actions 自动打包，workflow 文件在：

`.github/workflows/release.yml`

它会在 GitHub 的 Windows runner 上执行以下动作：

1. 拉取代码
2. 安装 Python 依赖
3. 使用 `PyInstaller --onefile` 打包
4. 自动创建 GitHub Release
5. 上传单文件 `circle_board.exe`

## 触发条件

当前不是每次 `push` 都打包。

只有当你 `push` 一个符合 `v*` 规则的 Git tag 时，才会触发自动打包和发布。

例如这些 tag 会触发：

- `v1.0.0`
- `v1.0.1`
- `v2.0`

这些不会触发：

- `1.0.0`
- `release-1`
- 普通 `git push origin main`

## 发布新版本

推荐流程：

```powershell
git add .
git commit -m "prepare release"
git push origin main
git tag v1.0.0
git push origin v1.0.0
```

执行完后，GitHub 会自动开始打包。

## 打包产物在哪里

打包完成后，到 GitHub 仓库页面查看：

`Releases`

你会看到对应版本号的 release，例如 `v1.0.0`，里面会有：

- `circle_board.exe`

这是最终可分发的单文件程序。

## 修改版本号

每次发布都要使用新的 tag。

例如：

```powershell
git tag v1.0.1
git push origin v1.0.1
```

不要反复使用同一个 tag，否则 release 创建会冲突。

## 注意事项

- 当前程序打开图片时，要求图片路径为纯英文目录，中文路径会提示不支持。
- 如果仓库是私有仓库，GitHub Actions 的 Windows 构建会消耗 Actions 分钟数。
- 如果你以后想加程序图标，可以在打包命令里继续加 `--icon your.ico`。

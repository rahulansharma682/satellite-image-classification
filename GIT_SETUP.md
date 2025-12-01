# GitHub Setup Guide

Quick guide to push this project to GitHub.

## Prerequisites

- Git installed on your system
- GitHub account created
- GitHub repository created (can be empty)

## Step-by-Step Instructions

### 1. Initialize Git Repository

```bash
cd "/mnt/c/Users/chrah/Desktop/Mini Projects/satellite-image-classification"
git init
```

### 2. Add Files to Git

```bash
# Add all files (gitignore will exclude unnecessary ones)
git add .

# Check what will be committed
git status
```

### 3. Create Initial Commit

```bash
git commit -m "Initial commit: Satellite Image Classification project with 98% accuracy"
```

### 4. Connect to GitHub Repository

Replace `yourusername` and `your-repo-name` with your actual GitHub username and repository name:

```bash
git remote add origin https://github.com/yourusername/your-repo-name.git

# Verify remote was added
git remote -v
```

### 5. Push to GitHub

```bash
# Push to main branch
git branch -M main
git push -u origin main
```

If you encounter authentication issues, you may need to:
- Use a Personal Access Token instead of password
- Or set up SSH keys

### 6. Verify on GitHub

Go to your GitHub repository URL and verify:
- ✅ README displays correctly with images
- ✅ Code is properly formatted
- ✅ Badges appear at the top
- ✅ Images load from assets/images/

## What Gets Pushed

Based on `.gitignore`, these files are **included**:
- ✅ README.md
- ✅ Jupyter notebook
- ✅ Visualizations in assets/images/
- ✅ Results JSON and summary
- ✅ LICENSE, requirements.txt, .gitignore

These files are **excluded**:
- ❌ EuroSAT dataset (too large)
- ❌ Model weights .pth files (large, unless you want them)
- ❌ Python cache files
- ❌ Temporary files

## Optional: Add Model Weights

If you want to include the trained model (~20MB):

1. Edit `.gitignore` and comment out:
   ```
   # *.pth
   ```

2. Add and commit:
   ```bash
   git add satellite_classification_results/satellite_classification_results/models/best_model.pth
   git commit -m "Add trained model weights"
   git push
   ```

## Optional: Use Git Large File Storage (LFS)

For large files like models, consider Git LFS:

```bash
# Install Git LFS
git lfs install

# Track .pth files
git lfs track "*.pth"

# Add and commit
git add .gitattributes
git add *.pth
git commit -m "Add model with Git LFS"
git push
```

## Updating Your Repository

After making changes:

```bash
git add .
git commit -m "Description of your changes"
git push
```

## Common Issues

### Authentication Failed
Use a Personal Access Token:
1. Go to GitHub Settings → Developer settings → Personal access tokens
2. Generate new token with `repo` scope
3. Use token as password when pushing

### Images Not Showing
- Verify images exist in `assets/images/` directory
- Check image paths in README.md match actual file locations
- GitHub may take a few seconds to display images after first push

### Large Files Error
If you get "file too large" errors:
- Remove large files from git: `git rm --cached largefile.pth`
- Add to .gitignore
- Commit and push again

## Pro Tips

1. **Add a description** to your GitHub repo for better SEO
2. **Add topics/tags**: machine-learning, deep-learning, pytorch, computer-vision, satellite-imagery
3. **Enable GitHub Pages** if you want to host a demo
4. **Add a star** to encourage others to do the same!

---

**Your repository is now ready to impress recruiters and collaborators!** 🚀

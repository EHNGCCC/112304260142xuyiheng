# GitHub Setup Guide

This guide is written for uploading this project to GitHub in a clean and traceable way.

## 1. Create the GitHub Repository
Recommended repository name:

```text
kaggle-imdb-sentiment-formal
```

Create an empty repository on GitHub first.

## 2. Configure Git on the Local Computer
Run these commands in PowerShell and replace them with your own information:

```bash
git config --global user.name "Your GitHub Name"
git config --global user.email "your_email@example.com"
```

Check the result:

```bash
git config --global --get user.name
git config --global --get user.email
```

## 3. Connect the Local Computer to GitHub
This computer already has an SSH key under:

```text
%USERPROFILE%\.ssh\id_rsa.pub
```

To view the public key in PowerShell:

```bash
Get-Content $HOME\.ssh\id_rsa.pub
```

Then copy its content and add it to GitHub:
- GitHub `Settings`
- `SSH and GPG keys`
- `New SSH key`

After that, test the SSH connection:

```bash
ssh -T git@github.com
```

## 4. Initialize This Repository Locally
Open PowerShell in the repository root and run:

```bash
git init
git branch -M main
git add .
git commit -m "Initialize Kaggle IMDB sentiment experiment repository"
```

## 5. Link to GitHub and Push
Replace the SSH URL with your real repository URL:

```bash
git remote add origin git@github.com:YOUR_USERNAME/YOUR_REPO.git
git push -u origin main
```

## 6. Update the Repository After Each Experiment
Every time you finish an experiment, update:
- `code/`
- `report/`
- `results/`

Then run:

```bash
git add .
git commit -m "Add cross-validation sparse rank blend experiment"
git push
```

Write clear commit messages. Avoid messages like `update` or `test`.

## 7. Recover a Better Old Version
To view history:

```bash
git log --oneline --decorate
```

To inspect an older result safely:

```bash
git switch -c revisit-best COMMIT_ID
```

To restore specific folders from an older commit:

```bash
git checkout COMMIT_ID -- code report results
```

This is safer than directly overwriting everything.

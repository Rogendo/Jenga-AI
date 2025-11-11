# Installation Guide

## Quick Setup

1. **Download the scripts** to your project root
2. **Run the setup**:
   ```bash
   chmod +x setup-git-automation.sh
   ./setup-git-automation.sh
   ```
3. **Restart your terminal**
4. **Test the commands**:
   ```bash
   git sc
   git etc "test commit"
   ```

## Manual Setup (if needed)

If the automated setup doesn't work:

1. Make scripts executable:
   ```bash
   chmod +x git-sc git-etc git-sync git-status-check
   ```

2. Add to your PATH manually or run from project directory

## Verification

Test if installation worked:
```bash
git sc
# Should show start session message
```

## Troubleshooting

- If commands not found, try sourcing your shell: `source ~/.bashrc`
- Check if scripts are in PATH: `echo $PATH`
- Ensure scripts are executable: `ls -la git-*`
- **Git Commands Not Recognized**: If you encounter an error like `git: 'sc' is not a git command` (or similar for `git etc`, `git sync`, `git status-check`), it means your shell hasn't correctly loaded the custom commands or the git aliases are not set up. To resolve this, you can manually create the git aliases:

  ```bash
  git config --global alias.sc '!sh -c "$HOME/bin/git-sc"'
  git config --global alias.etc '!sh -c "$HOME/bin/git-etc"'
  git config --global alias.sync '!sh -c "$HOME/bin/git-sync"'
  git config --global alias.status-check '!sh -c "$HOME/bin/git-status-check"'
  ```

  After running these commands, close and reopen your terminal or run `source ~/.bashrc` (or `source ~/.zshrc`) to ensure the changes are loaded in your current session.

## Usage Summary

1. **Start working**: `git sc`
2. **Make your changes**
3. **Finish working**: `git etc "your commit message"`

This automation will:
-  Automatically sync with main before starting work
-  Provide clear guidance at each step
-  Handle stashing/unstashing safely
-  Ensure proper commit messages
-  Minimize merge conflicts
-  Make collaboration seamless

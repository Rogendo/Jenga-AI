
# Jenga-AI Collaboration Guide

##  Overview
This document outlines the collaboration workflow, commit rules, and automation scripts for the Jenga-AI project to ensure smooth development and minimize conflicts.

##  Prerequisites
- Git installed on your machine
- SSH access to the repository
- Basic understanding of Git workflows

##  Setup & Installation

### 1. Clone the Repository
```bash
git clone git@github.com:Rogendo/Jenga-AI.git
cd Jenga-AI
```

### 2. Setup Automation Scripts
Run the setup script to install the custom Git commands:
```bash
chmod +x setup-git-automation.sh
./setup-git-automation.sh
```

##  Daily Workflow

### Starting Work Session - `git sc`
When you begin working, always start with:
```bash
git sc
```
This will:
- Pull latest changes from main branch
- Check for updates
- Ensure your local branch is up to date
- Provide guidance on current branch status

### Making Changes
1. Create a feature branch for significant changes:
```bash
git checkout -b feature/your-feature-name
```

2. Make your changes and test locally

### Ending Work Session - `git etc`
When you're done with your changes:
```bash
git etc "Your descriptive commit message"
```
This will:
- Stage all changes
- Commit with your message
- Push to remote repository
- Provide summary of actions taken

##  Commit Guidelines

### Commit Message Format
```
[scope] Brief description of changes

Detailed description if needed:
- What changed
- Why it changed
- Any breaking changes
```

### Examples:
```
[models] Updated BERT models to include new feature

- Added attention mechanisms to multitask_bert
- Improved training efficiency by 15%
- No breaking changes to existing APIs
```

```
[api] Fixed authentication bug

- Resolved issue with token expiration
- Added proper error handling
- Updated documentation
```

##  Branching Strategy

### Main Branches
- `main` - Production-ready code
- `develop` - Development integration branch

### Feature Branches
- `feature/` - New features
- `bugfix/` - Bug fixes
- `hotfix/` - Critical production fixes

### Naming Convention
```
feature/user-authentication
bugfix/login-validation
hotfix/critical-security-patch
```

##  Merge & Conflict Resolution

### Before Merging
1. Ensure all tests pass
2. Update your branch with latest main
3. Resolve any conflicts locally
4. Request code review

### Conflict Resolution Steps
1. Identify conflicting files
2. Communicate with team members
3. Resolve conflicts carefully
4. Test after resolution
5. Commit resolved changes

##  Automation Scripts

### Available Commands
- `git sc` - Start Commit (begin work session)
- `git etc "message"` - End Commit (complete work session)
- `git sync` - Sync with main branch
- `git status-check` - Comprehensive status check

### Manual Alternatives
If automation fails, use these manual commands:

**Start work:**
```bash
git pull origin main
git status
```

**End work:**
```bash
git add .
git commit -m "Your message"
git push origin current-branch
```

##  Best Practices

### Daily Routine
1. Start with `git sc`
2. Work on your feature/bugfix
3. Test thoroughly
4. End with `git etc "descriptive message"`
5. Create PR if feature complete

### Code Quality
- Write meaningful commit messages
- Keep commits focused and atomic
- Review code before committing
- Test before pushing

### Communication
- Notify team of major changes
- Discuss potential conflicts
- Update documentation
- Share progress in team channels

##  Emergency Procedures

### If Automation Fails
1. Don't panic - use manual commands
2. Contact team lead
3. Document the issue
4. Fix script when possible

### Rollback Procedures
```bash
# Revert last commit
git revert HEAD

# Reset to previous state (use carefully)
git reset --hard HEAD~1
```

##  Support

### Getting Help
1. Check this documentation first
2. Consult team members
3. Contact maintainers

### Common Issues & Solutions
- **Merge conflicts**: Use `git mergetool`
- **Authentication issues**: Verify SSH keys
- **Permission denied**: Check repository access

---




### Additional Utility Scripts

**Git Sync Script (`git-sync`):**
```bash
#!/bin/bash

# git-sync - Sync with main branch
echo "🔄 Syncing with main branch..."

current_branch=$(git branch --show-current)

# Stash changes
git stash push -m "Auto-stash before sync $(date)"

# Update main
git checkout main
git pull origin main

# Return to original branch and merge
if [ "$current_branch" != "main" ]; then
    git checkout $current_branch
    git merge main
    echo "✅ Synced $current_branch with main"
else
    echo "✅ Main branch is up to date"
fi

# Apply stashed changes
git stash pop

echo "🔄 Sync complete!"
```

**Git Status Check Script (`git-status-check`):**
```bash
#!/bin/bash

# git-status-check - Comprehensive status check
echo "📊 Comprehensive Git Status Check"

echo "🌿 Current branch:"
git branch --show-current

echo ""
echo "📋 Status:"
git status --short

echo ""
echo "📜 Recent commits:"
git log --oneline -5

echo ""
echo "🔄 Remote status:"
git remote -v

echo ""
echo "📦 Stash list:"
git stash list
```


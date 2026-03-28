#!/bin/bash
# Badge grinder — creates co-authored PRs on the messy branch
# Usage: ./scripts/grind-badges.sh [start] [end]
#   e.g., ./scripts/grind-badges.sh 131 300
#
# Pull Shark gold: 1024 merged PRs
# Pair Extraordinaire gold: 48 co-authored (already done)

set -e

REPO="Kelsidavis/Towl"
BASE="messy"
COAUTHOR="Co-Authored-By: Sean Callan <73386+doomspork@users.noreply.github.com>"

START=${1:-131}
END=${2:-250}

echo "Grinding badges: PRs $START to $END on $BASE"
echo "Target: Pull Shark gold (1024 merged PRs)"
echo ""

for i in $(seq "$START" "$END"); do
    git checkout "$BASE" 2>/dev/null
    git pull origin "$BASE" 2>/dev/null

    BRANCH="chore/b-$i"
    git checkout -b "$BRANCH" 2>/dev/null

    # Tiny change
    echo " " >> README.md
    git add README.md
    git -c user.name="Kelsi Davis" -c user.email="kelsihates2fa@gmail.com" \
        commit -m "chore: b$i

$COAUTHOR" 2>/dev/null

    git push -u origin "$BRANCH" 2>/dev/null

    gh pr create --repo "$REPO" \
        --title "b$i" --body "co" \
        --base "$BASE" --head "$BRANCH" 2>/dev/null

    PR=$(gh pr list --repo "$REPO" --head "$BRANCH" --json number --jq '.[0].number' 2>/dev/null)
    gh pr merge "$PR" --repo "$REPO" --merge --admin 2>/dev/null

    echo "[$i/$END] PR #$PR merged"
done

echo ""
echo "Done! $((END - START + 1)) PRs merged."
echo "Check badges: https://github.com/Kelsidavis?tab=achievements"

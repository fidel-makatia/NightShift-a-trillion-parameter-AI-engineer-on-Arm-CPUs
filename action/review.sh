#!/usr/bin/env bash
# NightShift PR reviewer. Sends the PR diff to a self-hosted OpenAI-compatible
# endpoint (Kimi K2 on Arm CPUs) and posts the review as a PR comment.
set -euo pipefail

: "${NS_ENDPOINT:?set endpoint}" "${NS_MODEL:?}" "${PR_NUMBER:?}" "${REPO:?}" "${GH_TOKEN:?}"
NS_TIER="${NS_TIER:-deep}"
NS_MAX_TOKENS="${NS_MAX_TOKENS:-2000}"

# 1. Fetch the diff (cap size so a huge PR can't blow the context window)
DIFF=$(gh pr diff "$PR_NUMBER" --repo "$REPO" | head -c 60000)

SYSTEM="You are NightShift, a senior engineer doing an overnight code review. \
Review the diff and list concrete issues as 'SEVERITY: problem — fix', most severe first. \
Cover correctness, security, error handling, and resource cleanup. Be specific and concise. \
If the diff is clean, say so."

# 2. Build the request safely with jq (no shell interpolation of the diff)
REQ=$(jq -n --arg sys "$SYSTEM" --arg diff "$DIFF" --arg model "$NS_MODEL" \
  --argjson maxtok "$NS_MAX_TOKENS" '{
    model: $model, max_tokens: $maxtok, temperature: 0.2,
    messages: [
      {role:"system", content:$sys},
      {role:"user",   content:("Review this pull request diff:\n\n```diff\n" + $diff + "\n```")}
    ]}')

# 3. Call the model, capture timings for the footer
RESP=$(curl -sS --fail-with-body "${NS_ENDPOINT%/}/chat/completions" \
  -H "Content-Type: application/json" -d "$REQ")

REVIEW=$(echo "$RESP" | jq -r '.choices[0].message.content // .choices[0].message.reasoning_content // "‹no content›"')
TPS=$(echo "$RESP" | jq -r '(.timings.predicted_per_second // 0) | (. * 10 | round / 10)')

# 4. Post it
BODY=$(cat <<EOF
### 🌙 NightShift review — \`${NS_MODEL}\` (${NS_TIER} tier)

${REVIEW}

<sub>Generated on self-hosted Arm CPUs (Azure Cobalt 100), no GPU · ${TPS} tok/s · your code never left your tenant.</sub>
EOF
)
gh pr comment "$PR_NUMBER" --repo "$REPO" --body "$BODY"
echo "NightShift review posted to $REPO#$PR_NUMBER (${TPS} tok/s)."

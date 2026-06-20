# Linux Auto-Trainer Setup Instructions

**Purpose:** Allow Mac to trigger Linux training through Dropbox (no SSH needed)

**How it works:**
1. Linux runs a cron job that checks for trigger file every minute
2. Mac creates trigger file in Dropbox
3. Linux detects trigger, updates paths, starts training
4. Linux creates confirmation file for Mac to see

---

## ONE-TIME SETUP ON LINUX MACHINE

### Step 1: Update Paths in Auto-Trainer Script

Edit the script to match your Linux environment:

```bash
cd ~/Dropbox/Conda/mynanet
nano linux_auto_trainer.sh
```

Update these lines (around line 18-19):
```bash
SPLITS_CSV="$HOME/Dropbox/Conda/SEAbird/seabird_validation/splits_csv/seabird_splits_80_10_10_seed42.csv"
FLAT_DIR="/mnt/data/seabird16khz_flat"  # ← CHANGE THIS TO YOUR DATA PATH
```

Save and exit (Ctrl+X, Y, Enter)

### Step 2: Make Script Executable

```bash
chmod +x ~/Dropbox/Conda/mynanet/linux_auto_trainer.sh
```

### Step 3: Test the Script Manually

```bash
# Run it once to verify it works
~/Dropbox/Conda/mynanet/linux_auto_trainer.sh

# Check status was created
cat ~/Dropbox/Conda/mynanet/LINUX_STATUS.txt
```

You should see:
```
Linux agent alive: [timestamp]
Hostname: [your-linux-machine]
GPU: [your GPU name]
Checking for trigger: .../START_TRAINING_NOW.trigger
```

### Step 4: Add to Crontab (Auto-run Every Minute)

```bash
# Open crontab editor
crontab -e
```

Add this line at the end:
```bash
* * * * * $HOME/Dropbox/Conda/mynanet/linux_auto_trainer.sh
```

Save and exit (in nano: Ctrl+X, Y, Enter; in vi: ESC, :wq, Enter)

### Step 5: Verify Cron is Working

```bash
# Wait 1-2 minutes, then check:
cat ~/Dropbox/Conda/mynanet/LINUX_STATUS.txt

# Timestamp should be recent (within last minute)
```

---

## USAGE FROM MAC (TONIGHT)

Once Linux is set up, triggering training from Mac is simple:

```bash
cd ~/Dropbox/Conda/mynanet
./TRIGGER_FROM_MAC.sh
```

The script will:
1. Check if Linux agent is alive
2. Create trigger file
3. Wait for confirmation (up to 2 minutes)
4. Show training started message

---

## MONITORING PROGRESS FROM MAC

### Check if Linux Detected Trigger
```bash
cat ~/Dropbox/Conda/mynanet/LINUX_STATUS.txt
```

### Check Training Started
```bash
cat ~/Dropbox/Conda/mynanet/TRAINING_CONFIRMED.txt
```

### Monitor Training Logs (Updates via Dropbox)
```bash
# Main orchestration log
tail -f ~/Dropbox/Conda/mynanet/authoritative_master.log

# Individual model logs
tail -f ~/Dropbox/Conda/mynanet/v1_seed42_linux.log
tail -f ~/Dropbox/Conda/mynanet/1e_seed42_linux.log
```

### Check Results (After ~3 hours per model)
```bash
# Count completed models
ls ~/Dropbox/Conda/mynanet/results_linux/*seed* -d 2>/dev/null | wc -l

# Check accuracies
grep "INT8 Accuracy:" ~/Dropbox/Conda/mynanet/results_linux/*/training_report.txt
```

---

## TROUBLESHOOTING

### "No status file found"
→ Linux agent not running. Check crontab setup on Linux.

### "Status file is old"
→ Cron job stopped or Linux machine offline.

### "No confirmation after 2 minutes"
→ Normal if Dropbox sync is slow. Check again in 5 minutes.

### Training doesn't start
→ Check paths in linux_auto_trainer.sh are correct for your Linux environment.

### How to stop training remotely
→ Cannot stop via Dropbox. Need physical access or SSH.
   Workaround: Training script is idempotent - can kill and restart.

---

## ADVANCED: Manual Trigger (Direct File Creation)

If the trigger script doesn't work, create trigger file manually:

```bash
# On Mac
touch ~/Dropbox/Conda/mynanet/START_TRAINING_NOW.trigger

# Wait for Dropbox sync + Linux cron (1-2 minutes)

# Check confirmation
cat ~/Dropbox/Conda/mynanet/TRAINING_CONFIRMED.txt
```

---

## SYSTEM OVERVIEW

```
┌─────────────────────────────────────────────────────────┐
│                         MAC                             │
│  1. Run: ./TRIGGER_FROM_MAC.sh                         │
│  2. Creates: START_TRAINING_NOW.trigger                │
│                       ↓ (Dropbox sync)                  │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│                    DROPBOX CLOUD                        │
│  Syncs files between Mac and Linux                     │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│                       LINUX                             │
│  1. Cron runs every minute: linux_auto_trainer.sh      │
│  2. Detects: START_TRAINING_NOW.trigger                │
│  3. Starts: run_linux_authoritative.sh                 │
│  4. Creates: TRAINING_CONFIRMED.txt                    │
│  5. Creates: LINUX_STATUS.txt (every minute)           │
│                       ↑ (Dropbox sync)                  │
└─────────────────────────────────────────────────────────┘
                          ↑
┌─────────────────────────────────────────────────────────┐
│                         MAC                             │
│  Monitor via synced files:                             │
│   - LINUX_STATUS.txt                                   │
│   - TRAINING_CONFIRMED.txt                             │
│   - authoritative_master.log                           │
│   - results_linux/                                     │
└─────────────────────────────────────────────────────────┘
```

---

## TIMELINE

**Tonight (Mac):**
1. Linux does one-time setup (5 minutes)
2. Mac runs trigger script (1 minute)
3. Wait for confirmation (1-2 minutes)

**Overnight:**
- Training runs on Linux (18 hours)
- Logs sync to Mac via Dropbox
- Can monitor progress from Mac

**Tomorrow:**
- Check results via Dropbox
- Analyze with Mac
- Update documentation

---

*Created: February 8, 2026*
*System: Dropbox-based remote triggering for NAT-protected Linux machine*

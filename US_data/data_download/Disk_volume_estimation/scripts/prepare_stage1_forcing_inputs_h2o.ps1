<#
.SYNOPSIS
    Transfer static setup files for Stage 1 forcing acquisition to h2o.

.DESCRIPTION
    Transfers only the required STATIC inputs (grid definition JSONs, optionally
    the CAMELSH shapefiles.7z) from the local Windows machine to h2o.

    Does NOT transfer any large generated forcing outputs (GRIB2, Parquet, NetCDF).
    Safe to run repeatedly — checks whether files already exist remotely before
    transferring.

    Uses the SSH host alias 'flashnh-h2o' (must be configured in ~/.ssh/config).
    No passwords are stored or embedded anywhere in this script.

.PARAMETER TransferShapefiles
    If set, also transfer shapefiles.7z to h2o for CAMELSH polygon weight build.
    Default: only grid definition JSONs are transferred.

.PARAMETER DryRun
    Print what would be transferred without actually running scp.

.PARAMETER H2oAlias
    SSH host alias for h2o (default: flashnh-h2o).
    Configure in ~/.ssh/config:
        Host flashnh-h2o
            HostName h2o.es.huji.ac.il
            User omrip
            IdentityFile ~/.ssh/id_ed25519_h2o

.EXAMPLE
    # Transfer only grid definition JSONs (minimum required for weight build):
    .\scripts\prepare_stage1_forcing_inputs_h2o.ps1

    # Also transfer CAMELSH shapefiles.7z (if not yet on h2o):
    .\scripts\prepare_stage1_forcing_inputs_h2o.ps1 -TransferShapefiles

    # Dry-run: print commands without executing:
    .\scripts\prepare_stage1_forcing_inputs_h2o.ps1 -DryRun
    .\scripts\prepare_stage1_forcing_inputs_h2o.ps1 -TransferShapefiles -DryRun

.NOTES
    CREDENTIAL POLICY: No passwords, tokens, or IP addresses are stored in this
    script. The SSH host alias resolves credentials from ~/.ssh/config only.

    After running, verify inputs on h2o:
        ssh flashnh-h2o "bash /path/to/repo/scripts/verify_stage1_forcing_inputs_h2o.sh"
#>

[CmdletBinding()]
param(
    [switch]$TransferShapefiles,
    [switch]$DryRun,
    [string]$H2oAlias = "flashnh-h2o"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# ---------------------------------------------------------------------------
# Local paths (relative to repo root)
# ---------------------------------------------------------------------------

$RepoRoot = Split-Path -Parent $PSScriptRoot
$LocalGridDefDir = Join-Path $RepoRoot "tmp\stage1_pilot_dryrun\09_manifests\stage1_pilot\grid_definitions"
$LocalMrmsGrid   = Join-Path $LocalGridDefDir "mrms_grid_definition.json"
$LocalRtmaGrid   = Join-Path $LocalGridDefDir "rtma_grid_definition.json"

# CAMELSH shapefile — look for shapefiles.7z under the local data root
$LocalShapefiles7z = $null
$CamelshSearchRoots = @(
    "C:\PhD\Python\neuralhydrology\US_data\data_download\Disk_volume_estimation\tmp\stage1_pilot_dryrun\02_basin_geometries\camelsh\shapefiles",
    "C:\PhD\Python\neuralhydrology\US_data\download",
    "$env:USERPROFILE\Downloads"
)
foreach ($searchRoot in $CamelshSearchRoots) {
    $candidate = Join-Path $searchRoot "shapefiles.7z"
    if (Test-Path $candidate) {
        $LocalShapefiles7z = $candidate
        break
    }
}

# ---------------------------------------------------------------------------
# Remote paths
# ---------------------------------------------------------------------------

$H2oForcingRoot = "/data42/omrip/Flash-NH/tmp/stage1_forcing_fullperiod"
$H2oGridDefDir  = "$H2oForcingRoot/grid_definitions"
$H2oCamelshDir  = "$H2oForcingRoot/02_basin_geometries/camelsh/shapefiles"

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

Write-Host "============================================================"
Write-Host "Flash-NH Stage 1 Forcing — Static Input Transfer"
Write-Host "============================================================"
Write-Host "SSH alias:        $H2oAlias"
Write-Host "Forcing root:     $H2oForcingRoot"
Write-Host "Grid def source:  $LocalGridDefDir"
Write-Host "Dry run:          $($DryRun.IsPresent)"
Write-Host "============================================================"
Write-Host ""

# ---------------------------------------------------------------------------
# Helper: run or print scp command
# ---------------------------------------------------------------------------

function Invoke-Scp {
    param(
        [string]$LocalPath,
        [string]$RemoteDir
    )

    $fileName = Split-Path -Leaf $LocalPath
    Write-Host "  Transferring: $fileName"
    Write-Host "    From: $LocalPath"
    Write-Host "    To:   ${H2oAlias}:${RemoteDir}/"

    if ($DryRun) {
        Write-Host "    [DRY RUN — not executed]"
        return $true
    }

    try {
        $result = & scp $LocalPath "${H2oAlias}:${RemoteDir}/" 2>&1
        if ($LASTEXITCODE -ne 0) {
            Write-Warning "  scp failed (exit $LASTEXITCODE): $result"
            return $false
        }
        Write-Host "    OK"
        return $true
    }
    catch {
        Write-Warning "  scp exception: $_"
        return $false
    }
}

function Test-RemoteFile {
    param([string]$RemotePath)
    if ($DryRun) { return $false }
    try {
        $out = & ssh $H2oAlias "test -f '$RemotePath' && echo exists || echo missing" 2>&1
        return ($out -match "exists")
    }
    catch { return $false }
}

function Invoke-RemoteMkdir {
    param([string]$RemoteDir)
    if ($DryRun) {
        Write-Host "    [DRY RUN] Would mkdir -p $RemoteDir on h2o"
        return
    }
    try {
        & ssh $H2oAlias "mkdir -p '$RemoteDir'" 2>&1 | Out-Null
    }
    catch {
        Write-Warning "  Could not create remote dir $RemoteDir : $_"
    }
}

# ---------------------------------------------------------------------------
# Transfer 1: MRMS grid definition JSON
# ---------------------------------------------------------------------------

Write-Host "[1] MRMS grid definition JSON"

if (-not (Test-Path $LocalMrmsGrid)) {
    Write-Warning "  LOCAL FILE MISSING: $LocalMrmsGrid"
    Write-Host "  This file is generated by the local pilot weight build (Milestone 2B)."
    Write-Host "  Run the local weight build first, then re-run this script."
}
else {
    $remoteTarget = "${H2oAlias}:${H2oGridDefDir}/mrms_grid_definition.json"
    $alreadyExists = Test-RemoteFile "$H2oGridDefDir/mrms_grid_definition.json"

    if ($alreadyExists -and -not $DryRun) {
        Write-Host "  SKIP: already exists on h2o at $H2oGridDefDir/mrms_grid_definition.json"
    }
    else {
        Invoke-RemoteMkdir $H2oGridDefDir
        $ok = Invoke-Scp -LocalPath $LocalMrmsGrid -RemoteDir $H2oGridDefDir
        if (-not $ok) {
            Write-Error "  Transfer failed for mrms_grid_definition.json"
        }
    }
}
Write-Host ""

# ---------------------------------------------------------------------------
# Transfer 2: RTMA grid definition JSON
# ---------------------------------------------------------------------------

Write-Host "[2] RTMA grid definition JSON"

if (-not (Test-Path $LocalRtmaGrid)) {
    Write-Warning "  LOCAL FILE MISSING: $LocalRtmaGrid"
    Write-Host "  This file is generated by the local pilot weight build (Milestone 2B)."
}
else {
    $alreadyExists = Test-RemoteFile "$H2oGridDefDir/rtma_grid_definition.json"

    if ($alreadyExists -and -not $DryRun) {
        Write-Host "  SKIP: already exists on h2o at $H2oGridDefDir/rtma_grid_definition.json"
    }
    else {
        Invoke-RemoteMkdir $H2oGridDefDir
        $ok = Invoke-Scp -LocalPath $LocalRtmaGrid -RemoteDir $H2oGridDefDir
        if (-not $ok) {
            Write-Error "  Transfer failed for rtma_grid_definition.json"
        }
    }
}
Write-Host ""

# ---------------------------------------------------------------------------
# Transfer 3: CAMELSH shapefiles.7z (optional, only if -TransferShapefiles)
# ---------------------------------------------------------------------------

Write-Host "[3] CAMELSH shapefiles.7z (for basin polygon weight build)"

if (-not $TransferShapefiles) {
    Write-Host "  SKIPPED — use -TransferShapefiles to enable."
    Write-Host ""
    Write-Host "  Before skipping, check if CAMELSH_shapefile.shp is already on h2o:"
    Write-Host "    ssh $H2oAlias ""find /data42 -name 'CAMELSH_shapefile.shp' 2>/dev/null"""
    Write-Host ""
    Write-Host "  If absent, re-run this script with -TransferShapefiles."
}
else {
    Write-Host "  CAMELSH shapefile — required for Milestone 2K-A weight build."
    Write-Host "  Note: pilot 50-basin weights are NOT valid for v001."
    Write-Host "        New 2,752-basin weight tables must be built from CAMELSH shapefile."
    Write-Host ""

    # Check if already on h2o
    $alreadyOnH2o = Test-RemoteFile "$H2oCamelshDir/CAMELSH_shapefile.shp"

    if ($alreadyOnH2o -and -not $DryRun) {
        Write-Host "  SKIP: CAMELSH_shapefile.shp already exists on h2o at $H2oCamelshDir"
    }
    else {
        # First check standard path (in case it's already extracted elsewhere)
        $alreadyAlt = $false
        if (-not $DryRun) {
            try {
                $findResult = & ssh $H2oAlias "find /data42 -name 'CAMELSH_shapefile.shp' 2>/dev/null | head -1" 2>&1
                if ($findResult -match "/") {
                    Write-Host "  Found on h2o at: $findResult"
                    Write-Host "  No transfer needed. Use --camelsh-polygons '$findResult' when building weights."
                    $alreadyAlt = $true
                }
            }
            catch {}
        }

        if (-not $alreadyAlt) {
            if ($null -eq $LocalShapefiles7z) {
                Write-Warning "  LOCAL FILE NOT FOUND: shapefiles.7z"
                Write-Host "  Searched in:"
                foreach ($sr in $CamelshSearchRoots) { Write-Host "    $sr" }
                Write-Host ""
                Write-Host "  Download from DOI 10.5281/zenodo.15066778:"
                Write-Host "    https://zenodo.org/records/15066778/files/shapefiles.7z"
                Write-Host "  Place the file in one of the search paths above, then re-run."
            }
            else {
                Write-Host "  Found local shapefiles.7z: $LocalShapefiles7z"
                $fileSizeMB = [math]::Round((Get-Item $LocalShapefiles7z).Length / 1MB, 1)
                Write-Host "  File size: ${fileSizeMB} MB (~506 MB expected)"

                Invoke-RemoteMkdir $H2oCamelshDir
                $ok = Invoke-Scp -LocalPath $LocalShapefiles7z -RemoteDir $H2oCamelshDir

                if ($ok -and -not $DryRun) {
                    Write-Host ""
                    Write-Host "  Next step on h2o — extract the archive:"
                    Write-Host "    ssh $H2oAlias """
                    Write-Host "    cd $H2oCamelshDir"
                    Write-Host "    7z x shapefiles.7z"
                    Write-Host "    ls -lh CAMELSH_shapefile.shp"
                    Write-Host "    """
                    Write-Host "  Expected result: $H2oCamelshDir/CAMELSH_shapefile.shp"
                }
            }
        }
    }
}
Write-Host ""

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

Write-Host "============================================================"
Write-Host "TRANSFER COMPLETE"
Write-Host "============================================================"
Write-Host ""
Write-Host "Next steps on h2o:"
Write-Host "  1. Verify all inputs are present:"
Write-Host "       bash scripts/verify_stage1_forcing_inputs_h2o.sh"
Write-Host ""
Write-Host "  2. Generate v001_basin_list.csv (if not done):"
Write-Host "       python scripts/export_v001_basin_list.py"
Write-Host ""
Write-Host "  3. Run Milestone 2K-A — build v001 weight tables:"
Write-Host "       python scripts/build_stage1_basin_weights.py \"
Write-Host "           --basin-list $H2oForcingRoot/v001_basin_list.csv \"
Write-Host "           --out-tag v001_2752 \"
Write-Host "           --data-root $H2oForcingRoot"
Write-Host ""
Write-Host "  4. Run Milestone 2K-B — smoke test:"
Write-Host "       bash scripts/run_stage1_forcing_smoke_h2o.sh"
Write-Host ""

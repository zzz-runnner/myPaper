[CmdletBinding()]
param(
    [string]$ProjectRoot = 'E:\paperMainWork',
    [string]$ResearchRoot = 'E:\paperMainWork-lpsm-validation'
)

# Local, bounded, read-only snapshot. No SSH, network checks or file writes.
$ErrorActionPreference = 'Stop'
$gitAvailable = $null -ne (Get-Command git -ErrorAction SilentlyContinue)

function Get-RepositorySnapshot {
    param([string]$RepositoryPath)
    $snapshot = [ordered]@{
        path = $RepositoryPath
        exists = (Test-Path -LiteralPath $RepositoryPath -PathType Container)
        branch = $null
        head = $null
        status = @()
        status_total_lines = 0
        error = $null
    }
    if (-not $snapshot.exists) { return [pscustomobject]$snapshot }
    if (-not $gitAvailable) {
        $snapshot.error = 'git is unavailable'
        return [pscustomobject]$snapshot
    }
    $branchLines = @(& git --no-optional-locks -C $RepositoryPath branch --show-current 2>&1)
    if ($LASTEXITCODE -ne 0) {
        $snapshot.error = (($branchLines | Select-Object -First 3) -join "`n")
        return [pscustomobject]$snapshot
    }
    $snapshot.branch = ($branchLines -join '')
    $headLines = @(& git --no-optional-locks -C $RepositoryPath rev-parse HEAD 2>&1)
    if ($LASTEXITCODE -ne 0) { throw "Cannot read HEAD: $RepositoryPath" }
    $snapshot.head = ($headLines -join '')
    $statusLines = @(& git --no-optional-locks -C $RepositoryPath status --short --untracked-files=normal 2>&1)
    if ($LASTEXITCODE -ne 0) { throw "Cannot read status: $RepositoryPath" }
    $snapshot.status_total_lines = $statusLines.Count
    $snapshot.status = @($statusLines | Select-Object -First 50 | ForEach-Object { "$_" })
    return [pscustomobject]$snapshot
}

$taskRoot = Join-Path $ProjectRoot 'hongwai_release_2026-10-22\private\长期任务文档'
$resultsRoot = Join-Path $ResearchRoot 'experiments\outputs\results'
$resultDirectories = @()
if (Test-Path -LiteralPath $resultsRoot -PathType Container) {
    $resultDirectories = @(Get-ChildItem -LiteralPath $resultsRoot -Directory |
        Sort-Object Name -Descending | Select-Object -First 20 -ExpandProperty Name)
}
$diagnosticRoot = Join-Path $resultsRoot '20261026_fusion_diagnostics'
$exitPath = Join-Path $diagnosticRoot 'diagnostic_exit_status.json'
$exitStatus = $null
if (Test-Path -LiteralPath $exitPath -PathType Leaf) {
    $exitStatus = Get-Content -LiteralPath $exitPath -Raw -Encoding UTF8 | ConvertFrom-Json
}

[ordered]@{
    collected_at = (Get-Date).ToString('o')
    scope = 'Local files and Git only; server occupancy and job status were not queried.'
    repositories = @((Get-RepositorySnapshot $ProjectRoot), (Get-RepositorySnapshot $ResearchRoot))
    current_document = [ordered]@{
        path = (Join-Path $taskRoot '项目长期状态-CURRENT.md')
        exists = (Test-Path -LiteralPath (Join-Path $taskRoot '项目长期状态-CURRENT.md') -PathType Leaf)
    }
    research_instructions_exist = (Test-Path -LiteralPath (Join-Path $ResearchRoot 'AGENTS.md') -PathType Leaf)
    results_root = $resultsRoot
    result_directories = $resultDirectories
    known_diagnostic = [ordered]@{
        path = $diagnosticRoot
        exit_status = $exitStatus
        report_exists = (Test-Path -LiteralPath (Join-Path $diagnosticRoot 'report.json') -PathType Leaf)
        download_index_exists = (Test-Path -LiteralPath (Join-Path $diagnosticRoot 'download_index.json') -PathType Leaf)
        note = 'Presence and recorded exit codes only; hashes were not reverified by this snapshot.'
    }
} | ConvertTo-Json -Depth 8

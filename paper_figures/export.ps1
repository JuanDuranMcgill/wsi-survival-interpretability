# Export one-slide figure decks to PDF (vector) and PNG (for checking) through PowerPoint.
# Usage: powershell -File export.ps1 fig1_ier_overview.pptx [more.pptx ...]
param([Parameter(ValueFromRemainingArguments = $true)][string[]]$Decks)
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$app = New-Object -ComObject PowerPoint.Application
try {
    foreach ($d in $Decks) {
        $full = Join-Path $here $d
        $base = [System.IO.Path]::ChangeExtension($full, $null).TrimEnd('.')
        $p = $app.Presentations.Open($full, $true, $false, $false)
        $p.SaveAs("$base.pdf", 32)
        $w = [int]($p.PageSetup.SlideWidth / 72 * 300)
        $h = [int]($p.PageSetup.SlideHeight / 72 * 300)
        $p.Slides.Item(1).Export("$base.png", "PNG", $w, $h)
        $p.Close()
        Write-Output "exported $base.pdf and .png ($w x $h)"
    }
} finally { $app.Quit() }

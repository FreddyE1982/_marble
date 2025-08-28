$repo = "https://github.com/FreddyE1982/_marble/"
$dir = "_marble"
if (Test-Path "$dir/.git") {
    git -C $dir pull --ff-only
} else {
    git clone $repo $dir
}
Set-Location $dir
try {
    py -3 -m pip install -e .
} catch {
    python -m pip install -e .
}

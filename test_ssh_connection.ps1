# SSH连接诊断脚本
param(
    [Parameter(Mandatory=$true)]
    [string]$Hostname,
    
    [Parameter(Mandatory=$false)]
    [string]$Username = "liuyuxuan",
    
    [Parameter(Mandatory=$false)]
    [int]$Port = 22
)

Write-Host "=== SSH连接诊断工具 ===" -ForegroundColor Cyan
Write-Host "目标: $Username@$Hostname:$Port" -ForegroundColor Yellow
Write-Host ""

# 1. 检查网络连通性
Write-Host "[1/4] 检查网络连通性..." -ForegroundColor Yellow
$pingResult = Test-Connection -ComputerName $Hostname -Count 2 -Quiet
if ($pingResult) {
    Write-Host "✓ 网络连通正常" -ForegroundColor Green
} else {
    Write-Host "✗ 网络不通，无法ping通目标主机" -ForegroundColor Red
    Write-Host "  可能原因:" -ForegroundColor Yellow
    Write-Host "  - 目标设备未开机或IP地址错误" -ForegroundColor Gray
    Write-Host "  - 不在同一网络" -ForegroundColor Gray
    Write-Host "  - 防火墙阻止了ICMP包" -ForegroundColor Gray
}

Write-Host ""

# 2. 检查SSH端口是否开放
Write-Host "[2/4] 检查SSH端口 ($Port) 是否开放..." -ForegroundColor Yellow
try {
    $tcpClient = New-Object System.Net.Sockets.TcpClient
    $connect = $tcpClient.BeginConnect($Hostname, $Port, $null, $null)
    $wait = $connect.AsyncWaitHandle.WaitOne(3000, $false)
    
    if ($wait) {
        $tcpClient.EndConnect($connect)
        Write-Host "✓ SSH端口 $Port 可访问" -ForegroundColor Green
        $tcpClient.Close()
    } else {
        Write-Host "✗ SSH端口 $Port 无法连接（超时）" -ForegroundColor Red
        $tcpClient.Close()
    }
} catch {
    Write-Host "✗ 无法连接到SSH端口: $_" -ForegroundColor Red
}

Write-Host ""

# 3. 显示本地网络信息
Write-Host "[3/4] 本地网络信息:" -ForegroundColor Yellow
$localIPs = Get-NetIPAddress -AddressFamily IPv4 | Where-Object {$_.IPAddress -notlike "127.*"} | Select-Object -ExpandProperty IPAddress
foreach ($ip in $localIPs) {
    Write-Host "  本地IP: $ip" -ForegroundColor Gray
}

Write-Host ""

# 4. 尝试SSH连接（详细模式）
Write-Host "[4/4] 尝试SSH连接（详细模式）..." -ForegroundColor Yellow
Write-Host "执行命令: ssh -v $Username@$Hostname -p $Port" -ForegroundColor Gray
Write-Host "（按Ctrl+C取消）" -ForegroundColor Gray
Write-Host ""

# 提供建议
Write-Host "=== 建议 ===" -ForegroundColor Cyan
if (-not $pingResult) {
    Write-Host "1. 确认目标设备已开机且IP地址正确" -ForegroundColor Yellow
    Write-Host "2. 检查是否在同一局域网（192.168.0.x）" -ForegroundColor Yellow
    Write-Host "3. 如果目标设备在其他网络，需要VPN或端口转发" -ForegroundColor Yellow
}
Write-Host "4. 如果端口不是22，使用: ssh -p <端口> $Username@$Hostname" -ForegroundColor Yellow
Write-Host "5. 检查目标服务器的SSH服务是否运行: sudo systemctl status ssh" -ForegroundColor Yellow


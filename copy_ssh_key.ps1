# SSH公钥复制脚本
# 使用方法: .\copy_ssh_key.ps1 -Username "your_username" -Hostname "remote_server_ip_or_domain"

param(
    [Parameter(Mandatory=$true)]
    [string]$Username,
    
    [Parameter(Mandatory=$true)]
    [string]$Hostname
)

$publicKey = Get-Content "$env:USERPROFILE\.ssh\id_rsa.pub"

Write-Host "正在将公钥复制到 $Username@$Hostname ..." -ForegroundColor Yellow
Write-Host "公钥内容: $publicKey" -ForegroundColor Cyan

# 使用SSH命令将公钥添加到远程服务器
$command = "mkdir -p ~/.ssh && chmod 700 ~/.ssh && echo '$publicKey' >> ~/.ssh/authorized_keys && chmod 600 ~/.ssh/authorized_keys"

ssh "$Username@$Hostname" $command

if ($LASTEXITCODE -eq 0) {
    Write-Host "公钥已成功添加到远程服务器！" -ForegroundColor Green
    Write-Host "现在可以尝试无密码登录: ssh $Username@$Hostname" -ForegroundColor Green
} else {
    Write-Host "操作失败，请检查用户名、服务器地址和网络连接" -ForegroundColor Red
}


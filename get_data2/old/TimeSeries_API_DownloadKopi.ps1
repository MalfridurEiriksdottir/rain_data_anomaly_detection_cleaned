Write-Host "Starting script execution..."
# Function to read and parse the configuration file


function Read-ConfigFile {
    param (
        [string]$configFilePath
    )

    $config = @{}

    try {
        # Check if the config file exists
        if (-not (Test-Path -Path $configFilePath)) {
            Write-Host "Configuration file not found: $configFilePath" -ForegroundColor Red
            return $null
        }

        # Read all lines from the config file
        $lines = Get-Content -Path $configFilePath

        foreach ($line in $lines) {
            # Skip empty lines and comments
            if ($line.Trim() -match '^\s*#' -or $line.Trim() -eq '') {
                continue
            }

            # Split each line into key and value
            $parts = $line -split '=', 2
            if ($parts.Count -eq 2) {
                $key = $parts[0].Trim()
                $value = $parts[1].Trim()
                
                # Convert boolean values properly
                if ($value -match '^(true|false)$') {
                    $value = [bool]::Parse($value.ToLower())
                }
                # Convert date values to datetime
                elseif ($key -like "time_start" -or $key -like "time_end") {
                    $value = [datetime]::ParseExact($value, 'yyyy-MM-dd HH:mm:ss', $null)
                }
                # Convert EPSG value to integer
                elseif ($key -like "epsg") {
                    $value = [int]$value
                }
                
                # Add key-value pair to the config hashtable
                $config[$key] = $value
            }
        }
        return $config
    }
    catch {
        Write-Host "Error reading configuration file: $($_.Exception.Message)" -ForegroundColor Red
        return $null
    }
}

# Function to perform login and retrieve the token
function Login-ToVeVaAPI {
    param (
        [string]$email,
        [string]$password
    )

    $loginUrl = "https://api.veva.live/account/login"
    $payload = @{
        "Email"    = $email
        "Password" = $password
    } | ConvertTo-Json

    try {
        $response = Invoke-RestMethod -Method Post -Uri $loginUrl -Headers @{ 'Content-Type' = 'application/json' } -Body $payload -ErrorAction Stop
        if ($response -and $response.token) {
            Write-Host "Login successful." -ForegroundColor Green
            return $response.token
        } else {
            Write-Host "Login failed or invalid response received." -ForegroundColor Red
            return $null
        }
    }
    catch {
        Write-Host "Error during login: $($_.Exception.Message)" -ForegroundColor Red
        return $null
    }
}

# Function to call the VeVa Radar Point Observations API and filter data to every 5th minute
function Get-VeVaRadarPointObservations {
    param (
        [datetime]$time_start,
        [datetime]$time_end,
        [int]$x,
        [int]$y,
        [int]$radar_id,
        [int]$elevation,
        [bool]$add_zero,
        [int]$epsg,
        [bool]$bias,
        [string]$token,
        [string]$output_dir
    )

    $baseUrl = "https://api.veva.live"
    
    # API Query
    $url = "$baseUrl/Radar/GetTimeSeries?fromUtc=$($time_start.ToString('o'))&toUtc=$($time_end.ToString('o'))&x=$x&y=$y&radarId=$radar_id&elevation=$elevation&AddZero=$add_zero&epsg=$epsg&doBiasAdjustment=$bias"

    $headers = @{
        'Authorization' = "Bearer $token"
        'Content-Type' = 'application/json'
    }

    try {
        Write-Host "Requesting data from VeVa API: $url"
        $response = Invoke-RestMethod -Method Get -Uri $url -Headers $headers -ErrorAction Stop
        
        # Check if the response contains 'values' and filter data
        if ($response.values) {
            $filteredData = @()
            
            foreach ($entry in $response.values) {
                # Extract the timestamp
                $timestamp = $entry.Time

                # Keep only timestamps where the minute is a multiple of 5 (00, 05, 10, 15, ...)
                # if ($timestamp -match "T\d{2}:(00|05|10|15|20|25|30|35|40|45|50|55):\d{2}Z$") {
                $filteredData += $entry
                # }
            }

            # Convert filtered data to CSV
            $filename = "VeVaRadar_X${x}_Y${y}_${current_time:yyyyMMdd_HHmm}.csv"
            $csvFilePath = "C:\Users\malfriduranna.eiriks\Desktop\rain_data_anomaly_detection-main\radar_data\$filename"
            # C:\Users\malfriduranna.eiriks\Desktop\rain_data_anomaly_detection-main\get_data\TimeSeries_API_DownloadKopi.ps1
            $tempCsvPath = $csvFilePath + ".tmp"

            # Write to a temporary file first
            $csvData = $filteredData | ConvertTo-Csv -NoTypeInformation
            $csvData | Out-File -FilePath $tempCsvPath -Encoding UTF8

            # Ensure all handles are closed before renaming
            Start-Sleep -Seconds 1
            Move-Item -Path $tempCsvPath -Destination $csvFilePath -Force
            Write-Host "Filtered data saved to CSV: $csvFilePath" -ForegroundColor Green
        } else {
            Write-Host "No data returned by the API." -ForegroundColor Yellow
        }
    }
    catch {
        Write-Host "Error occurred while fetching data: $($_.Exception.Message)" -ForegroundColor Red
    }
}

# Main script execution starts here

$scriptDirectory = Split-Path -Parent $MyInvocation.MyCommand.Path
$configFilePath = Join-Path -Path $scriptDirectory -ChildPath "config.txt"

$config = Read-ConfigFile -configFilePath $configFilePath

if (-not $config) {
    Write-Host "Failed to read configuration. Exiting." -ForegroundColor Red
    Read-Host -Prompt "Press Enter to exit"
    exit
}

$time_start = $config['time_start']
$time_end = $config['time_end']
$x = [int]$config['x']
$y = [int]$config['y']
$radar_id = [int]$config['radar_id']
$elevation = 0
$add_zero = $true
$epsg = [int]$config['epsg']
$bias = [bool]$config['bias']
$output_dir = $config['output_dir']

$email = [Environment]::GetEnvironmentVariable("veva_email", [System.EnvironmentVariableTarget]::Process)
$password = [Environment]::GetEnvironmentVariable("veva_password", [System.EnvironmentVariableTarget]::Process)

if (-not $email -or -not $password) {
    Write-Host "Environment variables veva_email or veva_password are not set. Exiting." -ForegroundColor Red
    Read-Host -Prompt "Press Enter to exit"
    exit
}

$token = Login-ToVeVaAPI -email $email -password $password

if (-not $token) {
    Write-Host "Failed to authenticate. Exiting." -ForegroundColor Red
    Read-Host -Prompt "Press Enter to exit"
    exit
}

Get-VeVaRadarPointObservations -time_start $time_start -time_end $time_end -x $x -y $y -radar_id $radar_id -elevation $elevation -add_zero $add_zero -epsg $epsg -bias $bias -token $token -output_dir $output_dir

# Read-Host -Prompt "Press Enter to exit"

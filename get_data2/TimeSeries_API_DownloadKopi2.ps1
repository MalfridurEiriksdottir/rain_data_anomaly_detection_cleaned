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

# Function to call the VeVa Radar Point Observations API
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
        [string]$output_dir # <--- Use this parameter!
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

        # Check if the response contains 'values'
        if ($response -and $response.values) {
            # --- No filtering applied (as per original commented code) ---
            $fetchedData = $response.values

            # --- Define Output Filename based on Python expectation ---
            # Using a consistent name pattern for Python to find
            $filename = "TimeSeries_${radar_id}_X${x}_Y${y}_temp.csv" # Added _temp suffix

            # --- Construct Output Path using $output_dir ---
            # Ensure output_dir exists
            if (-not (Test-Path -Path $output_dir -PathType Container)) {
                 Write-Host "Output directory '$output_dir' not found. Creating it." -ForegroundColor Yellow
                 try {
                     New-Item -Path $output_dir -ItemType Directory -Force | Out-Null
                 } catch {
                     Write-Host "Error creating output directory '$output_dir': $($_.Exception.Message). Saving to script directory instead." -ForegroundColor Red
                     $output_dir = $scriptDirectory # Fallback to script's directory
                 }
            }
            $tempCsvPath = Join-Path -Path $output_dir -ChildPath $filename
            # --- --- --- --- --- --- --- --- --- --- --- ---

            # Write data to the temporary file (OVERWRITE is fine here)
            # Make sure $fetchedData is not null or empty before converting
            if ($fetchedData) {
                $csvData = $fetchedData | ConvertTo-Csv -NoTypeInformation
                $csvData | Out-File -FilePath $tempCsvPath -Encoding UTF8 -Force # Use -Force to overwrite if exists
                Write-Host "Data saved to temporary CSV: $tempCsvPath" -ForegroundColor Green
            } else {
                Write-Host "API returned response structure, but 'values' array was empty or null." -ForegroundColor Yellow
                # Optional: Create an empty temp file so Python knows PS ran?
                # New-Item -Path $tempCsvPath -ItemType File -Force | Out-Null
                # Write-Host "Created empty temporary file: $tempCsvPath" -ForegroundColor Yellow
            }

        } else {
            Write-Host "No 'values' data returned by the API or response was empty/invalid." -ForegroundColor Yellow
            # Optional: Create empty temp file here too?
        }
    }
    catch {
        Write-Host "Error occurred while fetching/saving data: $($_.Exception.Message)" -ForegroundColor Red
        # Optional: Log the full error details
        # Write-Error $_.Exception
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

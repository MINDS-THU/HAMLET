@echo off
echo Starting script...

SETLOCAL ENABLEDELAYEDEXPANSION

REM Activate Conda environment
CALL conda activate base

REM Change to project directory
cd /d "D:\Research\COOPA"

REM Parameters
SET data_split=validation
SET random_seed=30
SET num_workers=8
SET log_dir=logs

IF NOT EXIST %log_dir% (
    mkdir %log_dir%
)

REM Model list
SET models="gpt-4.1" "claude-sonnet-4-20250514" "gemini-2.5-pro-preview-06-05" "Qwen/Qwen3-32B" "google/gemma-3-27b-it" "mistralai/Mistral-Small-3.1-24B-Instruct-2503"

REM Define skip pairs as space-separated values (or leave empty)
SET skip_pairs=

REM Iterate over gain_from_trade values
FOR %%G IN (False True) DO (
    SET "gain_from_trade=%%~G"

    IF "!gain_from_trade!"=="True" (
        SET "gft_flag=gft"
    ) ELSE (
        SET "gft_flag=ngft"
    )

    ECHO ===== Running with gain_from_trade=!gain_from_trade! =====

    REM Main model pair loop
    FOR %%B IN (%models%) DO (
        FOR %%S IN (%models%) DO (
            SET "buyer=%%~B"
            SET "seller=%%~S"
            SET "pair=!buyer!::!seller!"

            REM Check skip list
            ECHO !skip_pairs! | findstr /C:"!pair!" >nul
            IF !errorlevel! EQU 0 (
                ECHO Skipping !pair!
            ) ELSE (
                REM Sanitize model names for file/folder names
                SET "buyer_safe=!buyer:/=_!"
                SET "buyer_safe=!buyer_safe::=_!"
                SET "seller_safe=!seller:/=_!"
                SET "seller_safe=!seller_safe::=_!"

                SET "log_subdir=%log_dir%\!buyer_safe!_!seller_safe!_%data_split%_!gft_flag!"

                IF NOT EXIST "!log_subdir!" (
                    mkdir "!log_subdir!"
                )

                REM One top-level meta log per run
                SET "meta_log=!log_subdir!\meta.log"

                ECHO Running: !pair! (gain_from_trade=!gain_from_trade!)
                echo Running !pair! with gain_from_trade=!gain_from_trade! > "!meta_log!"
                python -m apps.bargaining.run ^
                    --buyer_model "!buyer!" ^
                    --seller_model "!seller!" ^
                    --data_split %data_split% ^
                    --random_seed %random_seed% ^
                    --num_workers %num_workers% ^
                    --gain_from_trade !gain_from_trade! ^
                    --log_dir "!log_subdir!" >> "!meta_log!" 2>&1

                ECHO Done with !pair! (gain_from_trade=!gain_from_trade!)
                echo Done with !pair! >> "!meta_log!"
            )
        )
    )
)

ECHO All experiments completed!
PAUSE
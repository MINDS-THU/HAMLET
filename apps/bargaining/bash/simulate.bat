@echo off

for /l %%i in (425,1,499) do (
    echo Running data_idx=%%i with buyer=1.2, seller=0.8
    python -m apps.bargaining.simulate --data_idx %%i --buyer_fraction 1.2 --seller_fraction 0.8

    echo Running data_idx=%%i with buyer=0.8, seller=1.2
    python -m apps.bargaining.simulate --data_idx %%i --buyer_fraction 0.8 --seller_fraction 1.2
)

echo All simulations complete.
pause
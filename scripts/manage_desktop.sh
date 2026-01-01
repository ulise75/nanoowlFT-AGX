#!/bin/bash
# Script to disable/enable graphical desktop on Jetson to save memory

ACTION=${1:-status}

case $ACTION in
    disable)
        echo "Disabling graphical desktop..."
        sudo systemctl set-default multi-user.target
        echo "✓ Desktop will be disabled on next boot"
        echo "  To apply now, run: sudo systemctl isolate multi-user.target"
        echo "  Or reboot: sudo reboot"
        ;;
    
    enable)
        echo "Enabling graphical desktop..."
        sudo systemctl set-default graphical.target
        echo "✓ Desktop will be enabled on next boot"
        echo "  To apply now, run: sudo systemctl isolate graphical.target"
        echo "  Or reboot: sudo reboot"
        ;;
    
    stop-now)
        echo "Stopping graphical desktop immediately..."
        sudo systemctl isolate multi-user.target
        echo "✓ Desktop stopped (temporary, will restart on reboot)"
        ;;
    
    start-now)
        echo "Starting graphical desktop immediately..."
        sudo systemctl isolate graphical.target
        echo "✓ Desktop started"
        ;;
    
    status)
        echo "Current boot target:"
        systemctl get-default
        echo ""
        echo "Current graphical session status:"
        systemctl is-active graphical.target
        echo ""
        echo "Usage:"
        echo "  $0 disable     - Disable desktop on boot"
        echo "  $0 enable      - Enable desktop on boot"
        echo "  $0 stop-now    - Stop desktop now (temporary)"
        echo "  $0 start-now   - Start desktop now"
        echo "  $0 status      - Show current status"
        ;;
    
    *)
        echo "Unknown action: $ACTION"
        echo "Use: disable, enable, stop-now, start-now, or status"
        exit 1
        ;;
esac

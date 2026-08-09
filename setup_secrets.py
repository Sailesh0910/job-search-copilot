#!/usr/bin/env python3
"""Setup script for Job Search Copilot secrets in Databricks.

This script creates a secret scope and stores required API credentials:
- Lakebase connection string
- Adzuna App ID
- Adzuna App Key

Usage:
    python setup_secrets.py
"""

from databricks.sdk import WorkspaceClient
import getpass
import sys


def create_secret_scope(w, scope_name):
    """Create a Databricks secret scope if it doesn't exist."""
    try:
        # Try to list secrets to check if scope exists
        w.secrets.list_secrets(scope=scope_name)
        print(f"✓ Secret scope '{scope_name}' already exists")
        return True
    except Exception as e:
        if "does not exist" in str(e):
            # Scope doesn't exist, create it
            try:
                w.secrets.create_scope(
                    scope=scope_name,
                    initial_manage_principal="users"
                )
                print(f"✓ Created secret scope '{scope_name}'")
                return True
            except Exception as create_error:
                print(f"✗ Failed to create scope: {create_error}")
                return False
        else:
            print(f"✗ Error checking scope: {e}")
            return False


def store_secret(w, scope, key, value, display_name):
    """Store a secret in the given scope."""
    try:
        w.secrets.put_secret(
            scope=scope,
            key=key,
            string_value=value
        )
        print(f"✓ Stored {display_name}")
        return True
    except Exception as e:
        print(f"✗ Failed to store {display_name}: {e}")
        return False


def main():
    """Main setup function."""
    print("="*60)
    print("Job Search Copilot - Secret Setup")
    print("="*60)
    print()
    
    # Initialize Databricks client
    try:
        w = WorkspaceClient()
        print("✓ Connected to Databricks workspace")
        print()
    except Exception as e:
        print(f"✗ Failed to connect to Databricks: {e}")
        sys.exit(1)
    
    # Define scope name
    scope_name = "job-copilot"
    
    # Step 1: Create secret scope
    print("Step 1: Creating secret scope...")
    if not create_secret_scope(w, scope_name):
        print("\nSetup failed. Please check your permissions.")
        sys.exit(1)
    print()
    
    # Step 2: Collect secrets from user
    print("Step 2: Please provide the following credentials:")
    print("-" * 60)
    print()
    
    # Lakebase connection string
    print("[1/3] Lakebase Connection String")
    print("Format: postgresql://user:password@host:port/database")
    lakebase_conn = getpass.getpass("Enter connection string (input hidden): ").strip()
    if not lakebase_conn:
        print("✗ Connection string cannot be empty")
        sys.exit(1)
    print()
    
    # Adzuna App ID
    print("[2/3] Adzuna App ID")
    adzuna_app_id = input("Enter App ID: ").strip()
    if not adzuna_app_id:
        print("✗ App ID cannot be empty")
        sys.exit(1)
    print()
    
    # Adzuna App Key
    print("[3/3] Adzuna App Key")
    adzuna_app_key = getpass.getpass("Enter App Key (input hidden): ").strip()
    if not adzuna_app_key:
        print("✗ App Key cannot be empty")
        sys.exit(1)
    print()
    
    # Step 3: Store all secrets
    print("Step 3: Storing secrets...")
    print("-" * 60)
    
    success_count = 0
    
    if store_secret(w, scope_name, "LAKEBASE_CONNECTION_STRING", lakebase_conn, "Lakebase connection string"):
        success_count += 1
    
    if store_secret(w, scope_name, "ADZUNA_APP_ID", adzuna_app_id, "Adzuna App ID"):
        success_count += 1
    
    if store_secret(w, scope_name, "ADZUNA_APP_KEY", adzuna_app_key, "Adzuna App Key"):
        success_count += 1
    
    print()
    print("="*60)
    if success_count == 3:
        print("✓ Setup completed successfully!")
        print()
        print("Next: add all three as resources in the Databricks Apps UI, matching")
        print("the valueFrom keys already in app/app.yaml — the app reads them via")
        print(f"os.environ, not dbutils.secrets.get(), so this scope ('{scope_name}')")
        print("only needs to be the source the app resource points at.")
    else:
        print(f"⚠ Setup completed with errors ({success_count}/3 secrets stored)")
    print("="*60)


if __name__ == "__main__":
    main()
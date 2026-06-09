"""
Milvus rebuild script (calling core common tools)

Implemented based on methods provided by MilvusCollectionBase:
- Find the corresponding Collection management class by alias
- Call create_new_collection() to create a new collection (automatically create index and load)
- Perform data migration (supports batch processing to avoid memory overflow)
- Call switch_alias() to switch the alias to the new collection
- Optionally delete the old Collection

Note: This script migrates data by default (in batches of 3000).
To disable data migration, use the --no-migrate-data option.
"""

import argparse
import sys
import traceback
from typing import Optional, List

from pymilvus import Collection

from core.observation.logger import get_logger
from core.oxm.milvus.migration.utils import rebuild_collection


logger = get_logger(__name__)


def migrate_data_callback(
    old_collection: Collection, new_collection: Collection, batch_size: int = 3000
) -> None:
    """
    Data migration callback function (using offset pagination + sorting to ensure data integrity)

    Args:
        old_collection: Old collection instance
        new_collection: New collection instance
        batch_size: Number of records processed per batch, default is 3000

    Note:
        Use offset + limit + order_by for paginated queries to avoid:
        1. Data loss (unordered queries may return in unpredictable order)
        2. Data duplication (pagination position may drift)

        Although queries with large offsets are less efficient, they are acceptable for one-time data migration,
        and ensure data completeness and accuracy.
    """
    logger.info(
        "Start migrating data: %s -> %s (batch size: %d)",
        old_collection.name,
        new_collection.name,
        batch_size,
    )

    total_migrated = 0  # Total number of records migrated
    offset = 0  # Current query offset

    try:
        while True:
            logger.info(
                "Querying batch %d, offset: %d, limit: %d",
                total_migrated // batch_size + 1,
                offset,
                batch_size,
            )

            # Use offset+limit for pagination, and sort by id
            # Note: STL_SORT index on the id field is required to use order_by
            # Without an index, it may raise an error or have poor performance
            try:
                # Try using the order_by parameter (pymilvus 2.4+)
                query_result = old_collection.query(
                    expr="",  # Query all data
                    output_fields=["*"],
                    limit=batch_size,
                    offset=offset,
                    order_by=[("id", "asc")],  # Sort by id in ascending order
                )
            except TypeError:
                # If order_by is not supported (older version), fall back to unordered query
                # In this case, there's still a risk of data loss or duplication
                logger.warning(
                    "Current pymilvus version does not support order_by parameter, using unordered query"
                )
                logger.warning(
                    "It is recommended to upgrade pymilvus to version 2.4+, or create an STL_SORT index on the id field"
                )
                query_result = old_collection.query(
                    expr="", output_fields=["*"], limit=batch_size, offset=offset
                )
            except Exception as e:
                # If the error is due to missing index, prompt user to create one
                if "index" in str(e).lower() or "sort" in str(e).lower():
                    logger.error(
                        "Query failed, possibly because there is no STL_SORT index on the id field: %s",
                        e,
                    )
                    logger.error(
                        "Please create an STL_SORT index on the id field of the old collection, or use unordered query"
                    )
                raise

            # If query result is empty, no more data to migrate
            if not query_result:
                logger.info("No more data, migration completed")
                break

            # Get the number of records in current batch
            batch_count = len(query_result)
            logger.info(
                "Retrieved %d records, starting to insert into new collection...",
                batch_count,
            )

            # Insert into new collection
            new_collection.insert(query_result)
            new_collection.flush()

            # Update statistics
            total_migrated += batch_count
            offset += batch_count  # Update offset
            logger.info("Migrated %d records", total_migrated)

            # If the number of records retrieved is less than batch_size, it's the last batch
            if batch_count < batch_size:
                logger.info("Last batch, migration completed")
                break

    except Exception as e:
        logger.error("Error occurred during data migration: %s", e)
        raise

    logger.info("Data migration completed: total %d records", total_migrated)


def run(alias: str, drop_old: bool, migrate_data: bool, batch_size: int) -> None:
    """
    Execute rebuild logic (delegated to core tools)

    Args:
        alias: Collection alias
        drop_old: Whether to delete the old collection
        migrate_data: Whether to migrate data
        batch_size: Number of records processed per batch
    """
    try:
        # Determine whether to pass the callback function based on whether data migration is needed
        if migrate_data:
            # Wrap migrate_data_callback with lambda to pass batch_size parameter
            populate_fn = lambda old_col, new_col: migrate_data_callback(
                old_col, new_col, batch_size
            )
        else:
            populate_fn = None

        result = rebuild_collection(
            alias=alias, drop_old=drop_old, populate_fn=populate_fn
        )

        logger.info(
            "Milvus rebuild completed: alias=%s, src=%s -> dest=%s, dropped_old=%s",
            result.alias,
            result.source_collection,
            result.dest_collection,
            result.dropped_old,
        )
    except Exception as exc:
        logger.error("Milvus rebuild failed: %s", exc)
        traceback.print_exc()
        raise


def main(argv: Optional[List[str]] = None) -> int:
    """
    Main function: parse command-line arguments and execute rebuild

    Args:
        argv: List of command-line arguments

    Returns:
        Exit code (0 indicates success)
    """
    parser = argparse.ArgumentParser(
        description="Rebuild and switch Milvus Collection alias",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  # Rebuild collection and migrate data (default batch size 3000)
  python milvus_rebuild_collection.py -a episodic_memory
  
  # Rebuild collection without migrating data
  python milvus_rebuild_collection.py -a episodic_memory --no-migrate-data
  
  # Rebuild collection, migrate data and specify batch size
  python milvus_rebuild_collection.py -a episodic_memory --batch-size 5000
  
  # Rebuild collection, migrate data and delete old collection
  python milvus_rebuild_collection.py -a episodic_memory --drop-old
        """,
    )

    parser.add_argument(
        "--alias", "-a", required=True, help="Collection alias, e.g.: episodic_memory"
    )
    parser.add_argument(
        "--drop-old",
        "-x",
        action="store_true",
        help="Whether to delete old collection (default: keep)",
    )
    parser.add_argument(
        "--no-migrate-data",
        action="store_true",
        help="Do not migrate data (default: migrate data)",
    )
    parser.add_argument(
        "--batch-size",
        "-b",
        type=int,
        default=3000,
        help="Number of records per migration batch (default: 3000)",
    )

    args = parser.parse_args(argv)

    run(
        alias=args.alias,
        drop_old=args.drop_old,
        migrate_data=not args.no_migrate_data,  # Migrate data by default
        batch_size=args.batch_size,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

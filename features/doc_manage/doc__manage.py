from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, delete, and_, func
from uuid import UUID
from typing import Optional, List
from database.database_models import documents
from database.vector_database import pinecone_service

class DocumentRepository:
    """Document repository with dependency injection"""
    
    def __init__(self, session: AsyncSession):
        self.session = session
    
    async def get_document_by_id(self, document_id: UUID, user_id: UUID) -> Optional[documents]:
        """Get document by ID and user ID"""
        query = select(documents).where(
            and_(
                documents.id == document_id,
                documents.user_id == user_id
            )
        )
        result = await self.session.execute(query)
        return result.scalar_one_or_none()
    
    async def get_user_documents(self, user_id: UUID) -> List[documents]:
        """Get all documents for a user"""
        query = select(documents).where(
            documents.user_id == user_id
        ).order_by(documents.created_at.desc())
        result = await self.session.execute(query)
        return result.scalars().all()
    
    async def delete_document(self, document_id: UUID, user_id: UUID) -> Optional[documents]:
        """Delete a document by ID and user ID"""
        document = await self.get_document_by_id(document_id, user_id)
        if not document:
            return None
        delete_query = delete(documents).where(
            and_(
                documents.id == document_id,
                documents.user_id == user_id
            )
        )
        pinecone_service.delete(str(user_id), str(document_id))
        await self.session.execute(delete_query)
        await self.session.commit()
        
        return document
    
    async def count_user_documents(self, user_id: UUID) -> int:
        """Count total documents for a user"""
        query = select(func.count()).select_from(documents).where(documents.user_id == user_id)
        result = await self.session.execute(query)
        return result.scalar() or 0
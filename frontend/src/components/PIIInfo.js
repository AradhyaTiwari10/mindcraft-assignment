import React from 'react';

function PIIInfo({ detectedPII }) {
  if (!detectedPII || detectedPII.length === 0) return null;
  return (
    <div className="mt-4 bg-yellow-50 border-l-4 border-yellow-400 p-4">
      <h4 className="font-semibold mb-2">Detected PII</h4>
      <ul className="list-disc ml-6">
        {detectedPII.map((item, idx) => (
          <li key={idx}>
            <b>{item.type}:</b> {item.text}
          </li>
        ))}
      </ul>
    </div>
  );
}

export default PIIInfo; 
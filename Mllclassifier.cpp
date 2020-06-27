#include "stdafx.h"
#include "Mllclassifier.h"

namespace mll
{
	const bool mllclassifier::findSectionName(std::ifstream& reader, const string sectionName)
	{
		// Find section name
		string lineStr;
		string trimStr;
		bool findFlag = false;
		while (!reader.eof())
		{
			// Compare a string value
			getline(reader, lineStr);
			trimStr = trim(lineStr);
			if (trimStr == sectionName)
			{
				findFlag = true;
				break;
			}
		}

		return findFlag;
	}

	const string mllclassifier::getSectionName(const string name, const string prefix)
	{
		// Check the prefix
		if (prefix == "" || prefix.empty() == true)
		{
			return "[" + name + "]";
		}
		else
		{
			return "[" + prefix + "_" + name + "]";
		}
	}
}
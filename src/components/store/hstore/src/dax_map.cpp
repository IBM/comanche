/*
 * (C) Copyright IBM Corporation 2018, 2019. All rights reserved.
 * US Government Users Restricted Rights - Use, duplication or disclosure restricted by GSA ADP Schedule Contract with IBM Corp.
 */

#include "dax_map.h"

#include <rapidjson/schema.h>
#include <rapidjson/error/en.h>

#include <stdexcept>
#include <string>
#include <vector>

/*
 * The schema for the JSON "dax_map" parameter, in Draft 7 form as
 * described at https://json-schema.org:
 *
 * {
 *   "type": "array",
 *   "items": {
 *     "type": "object",
 *     "properties": {
 *       "numa_node": { "type": "integer", "minimum": 0 },
 *       "path": { "type": "string" },
 *       "addr": { "type": "integer", "minimum": 0 },
 *     },
 *     "required" : [ "numa_node", "path", "addr" ]
 *   }
 * }
 *
 */
namespace
{
	std::string error_report(const std::string &prefix, const std::string &text, const rapidjson::Document &doc)
	{
		return prefix + " '" + text + "': " + rapidjson::GetParseError_En(doc.GetParseError()) + " at " + std::to_string(doc.GetErrorOffset());
	}

	rapidjson::SchemaDocument make_schema_doc()
	{
		static const char *dax_map_schema =
			"{\n"
				"\"type\": \"array\",\n"
				"\"items\": {\n"
					"\"type\": \"object\",\n"
					"\"properties\": {\n"
						"\"numa_node\": { \"type\": \"integer\", \"minimum\": 0 },\n"
						"\"path\": { \"type\": \"string\" },\n"
						"\"addr\": { \"type\": \"integer\", \"minimum\": 0 }\n"
					"},\n"
					"\"required\" : [ \"numa_node\", \"path\", \"addr\" ]\n"
				"}\n"
			"}\n"
			;
		rapidjson::Document doc;
		doc.Parse(dax_map_schema);
		if ( doc.HasParseError() )
		{
			throw std::logic_error(error_report("Bad JSON dax_map_schema", dax_map_schema, doc));
		}
		return rapidjson::SchemaDocument(doc);
	}

	using json_value = const rapidjson::Value &;

	/*
	 * map of strings to enumerated values
	 */
	template <typename T>
		using translate_map = const std::map<std::string, T>;

	/*
	 * map of strings to json parse functions
	 */
	template <typename S>
		using parse_map = translate_map<void (*)(const json_value, S *)>;

	template <typename V>
		V parse_scalar(json_value &v);

	template <>
		int parse_scalar<int>(json_value &v)
		{
			if ( ! v.IsInt() )
			{
				throw std::domain_error("not an int");
			}
			return v.GetInt();
		}

	template <>
		std::uint64_t parse_scalar<std::uint64_t>(json_value &v)
		{
			if ( ! v.IsUint64() )
			{
				throw std::domain_error("not a uint64");
			}
			return v.GetUint64();
		}

	template <>
		std::string parse_scalar<std::string>(json_value &v)
		{
			if ( ! v.IsString() )
			{
				throw std::domain_error("not a string");
			}
			return v.GetString();
		}

	/* assignment */

	template <typename V>
		class Assign
		{
		public:
			template <typename S, V S::*M>
			static void assign_scalar(json_value &v, S *s)
			{
				s->*M = parse_scalar<V>(v);
			}
		};

#define SET_SCALAR(S,M) Assign<decltype(S::M)>::assign_scalar<S, &S::M>

	parse_map<nupm::Devdax_manager::config_t> config_t_attr
	{
		{ "numa_node", SET_SCALAR(nupm::Devdax_manager::config_t,numa_node) },
		{ "path", SET_SCALAR(nupm::Devdax_manager::config_t,path) },
		{ "addr", SET_SCALAR(nupm::Devdax_manager::config_t, addr) },
	};

	std::vector<nupm::Devdax_manager::config_t> parse_devdax_string(const std::string &dax_map_)
	{
		std::vector<nupm::Devdax_manager::config_t> dax_config;
		rapidjson::Document doc;
		{
			doc.Parse(dax_map_.c_str());
			if ( doc.HasParseError() )
			{
				throw std::domain_error(error_report("Bad JSON dax_map", dax_map_, doc));
			}
		}

		auto schema_doc(make_schema_doc());
		rapidjson::SchemaValidator validator(schema_doc);

		if ( ! doc.Accept(validator) )
		{
			{
				rapidjson::StringBuffer sb;
				validator.GetInvalidSchemaPointer().StringifyUriFragment(sb);
				std::cerr << "Invalid schema: " << sb.GetString() << "\n";
				std::cerr << "Invalid keyword: " << validator.GetInvalidSchemaKeyword() << "\n";
			}

			{
				rapidjson::StringBuffer sb;
				validator.GetInvalidDocumentPointer().StringifyUriFragment(sb);
				std::cerr << "Invalid document: " << sb.GetString() << "\n";
			}
			throw std::domain_error(error_report("JSON dax_map failed validation", dax_map_, doc));
		}

		for ( const auto & it : doc.GetArray() )
		{
			dax_config.emplace_back();
			for ( const auto & itr : it.GetObject() )
			{
				auto k = itr.name.GetString();
				auto iv = config_t_attr.find(k);
				try
				{
					if ( iv == config_t_attr.end() )
					{
						throw std::domain_error(": unrecognized key");
					}
					(iv->second)(itr.value, &dax_config.back());
				}
				catch (const std::domain_error &e)
				{
					throw std::domain_error{std::string{k} + " " + e.what()};
				}
			}
		}
		return dax_config;
	}
}

Devdax_manager::Devdax_manager(
	const std::string &dax_map
	, bool force_reset
)
	: nupm::Devdax_manager(parse_devdax_string(dax_map), force_reset)
{}
